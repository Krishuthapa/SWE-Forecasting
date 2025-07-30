import os
import sys

import torch
import torch.nn as nn
from torch import optim

import numpy as np
import pandas as pd

import time
import joblib

from model import SWETransformer
import transformer_dl as DL

from helper_functions import getDevice, getInputs, getFutureOutputs, getPastOutputs


############################################### All imports end here ###############################################

device = torch.device(getDevice())

BASE_URL = os.getenv("BASE_URL")

FORECASTING_WINDOW = int(os.getenv("FORECASTING_WINDOW"))
FORECASTING_STEP_SIZE = int(os.getenv("FORECASTING_STEP"))

############################################# Environment Variables ################################################

snotel_locations_info = pd.read_csv('{}/data/snotel_location_info_2.csv'.format(BASE_URL))

test_indices = [43, 66, 338, 463, 115, 105, 83, 306, 481, 21, 185, 136, 110, 406, 408, 132, 
                120, 197, 7, 301, 375, 236, 269, 287, 49, 241, 212, 145, 264, 435, 73, 277, 364, 
                438, 372, 368, 243, 493, 497, 14, 32, 433, 313, 213, 42, 475, 160, 190, 281, 285, 
                316, 461, 154, 424, 310, 340, 179, 167, 193, 466, 56, 158, 365, 492, 432, 415, 68, 
                409, 76, 148, 267, 227, 91, 189, 472, 162, 147, 395, 113, 2, 271, 357, 347, 335, 507, 
                465, 57, 19, 442, 381, 398, 79, 337, 501, 41, 339, 331, 63, 359, 232, 323, 258]

test_indices = sorted(test_indices)

######################################### Data loader to call all the useful data.##################################
dataloader = DL.DataLoader(test_indices = test_indices, temp_window = 7)

all_locations_distances, all_locations_angularities = dataloader.getDistanceAndAngularity()
model_inputs, model_outputs, all_dates = dataloader.getAllInputAndTargets()

train_loc_indices = dataloader.getTrainingIndices()


# Model Inputs and Outputs.
model_inputs = torch.from_numpy(model_inputs)
model_outputs = torch.from_numpy(model_outputs)

max_swe,min_swe = dataloader.getMaxMinSWE()
####################################################################################################################

swe_model = SWETransformer(model_dim = 512, enc_inp_dim = 21, dec_inp_dim = 1, nhead = 8, n_enc_layers = 3, 
                           n_dec_layers = 1, ffn_dim = 2048, output_dim = 1, forecasting_window = FORECASTING_WINDOW)

swe_model = swe_model.to(device=device)

# Loss Function 
mse_error = nn.MSELoss()

hasFile = os.path.isfile("{}/checkpoints/forecasting_daily_tf_model.pt".format(BASE_URL))

if hasFile:
    checkpoint = torch.load("{}/checkpoints/forecasting_daily_tf_model.pt".format(BASE_URL))
            
    epoch_number = checkpoint['epoch']
    strike_count = checkpoint['strike_count']
    swe_model.load_state_dict(checkpoint['swe_model_dict'])

testing_yr_indices_1 , testing_yr_indices_2, testing_yr_indices_3, testing_yr_indices_4, testing_yr_indices_5  = dataloader.getTestingDataIndices()
all_test_indices = list(testing_yr_indices_1) + list(testing_yr_indices_2) +list(testing_yr_indices_3) + list(testing_yr_indices_4) + list(testing_yr_indices_5)

def TestModelSP(testing_yr_indices, all_inputs, all_outputs, all_dates, batch_size = 128):
    mae_error = nn.MSELoss()
    
    actual_outputs = []
    predicted_outputs = []

    all_indices = sorted(list(train_loc_indices) + list(test_indices))

    with torch.no_grad():
        for index , input_index in enumerate(testing_yr_indices):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            instance_daily_input  = getInputs(all_inputs, input_index, all_indices, short_window = 365, step_size = FORECASTING_STEP_SIZE)
            instance_output = getFutureOutputs(all_outputs, input_index, all_indices, window = FORECASTING_WINDOW, step_size = FORECASTING_STEP_SIZE)

            instance_daily_input = instance_daily_input.to(device=device,dtype=torch.float32)
            instance_output = instance_output.to(device=device,dtype=torch.float32)

            encoder_x = instance_daily_input.to(device=device, dtype=torch.float32)
        
            decoder_past_x = getPastOutputs(all_outputs, input_index, all_indices, window = 45, step_size = FORECASTING_STEP_SIZE)
            decoder_past_x = decoder_past_x.to(device=device, dtype=torch.float32)
            decoder_past_x = ((decoder_past_x - min_swe)/(max_swe - min_swe))

            output_accumulator = []

            for loc_index in range(0, encoder_x.shape[0], batch_size):
                batch_encoder_x = encoder_x[loc_index:loc_index+batch_size,:,:]
                batch_decoder_x = decoder_past_x[loc_index:loc_index+batch_size,:,:]

                for auto_reg_index in range(FORECASTING_WINDOW):
                    tgt_len = batch_decoder_x.shape[1]
                    decoder_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device), diagonal=1).bool()

                    output = swe_model(batch_encoder_x, batch_decoder_x, decoder_attn_mask=decoder_mask)

                    selected_output = output[:, -1:, :]
                    batch_decoder_x = torch.cat((batch_decoder_x, selected_output), dim=1)

                output_accumulator.append(batch_decoder_x[:,-FORECASTING_WINDOW:,:])

            instance_output = instance_output.detach().cpu()
            final_output = torch.cat(output_accumulator,dim = 0).detach().cpu() * max_swe
                
            output_loss = mae_error(final_output.reshape(-1,FORECASTING_WINDOW), instance_output.reshape(-1,FORECASTING_WINDOW))            

            actual_outputs.append(instance_output.reshape(1,-1,FORECASTING_WINDOW).cpu().detach().numpy())
            predicted_outputs.append(final_output.reshape(1,-1,FORECASTING_WINDOW).cpu().numpy())
            
            print('Batch number :{} , actual_output_shape: {}, predicted_output_shape: {}, loss : {}'.format(index+1, instance_output.reshape(1,-1,FORECASTING_WINDOW).cpu().detach().numpy().shape, final_output.reshape(1,-1,FORECASTING_WINDOW).cpu().numpy().shape, output_loss.item()))

    actual_outputs = np.array(actual_outputs).reshape(-1,len(all_indices),FORECASTING_WINDOW)
    predicted_outputs = np.array(predicted_outputs).reshape(-1,len(all_indices),FORECASTING_WINDOW)
    
    return (actual_outputs, predicted_outputs)

def getDailyPredictions(data_indices, all_inputs,all_outputs,all_dates, batch_size = 128):
    actual_outputs = []
    collected_forecasted_outputs = []

    mse_loss = nn.MSELoss()

    all_indices = sorted(list(train_loc_indices) + list(test_indices))

    with torch.no_grad():

        for data_index in range(0,len(data_indices), FORECASTING_WINDOW):
            torch.cuda.empty_cache()

            instance_daily_input  = getInputs(all_inputs, data_indices[data_index], all_indices, short_window = 365, step_size = FORECASTING_STEP_SIZE)
            instance_output = getFutureOutputs(all_outputs, data_indices[data_index], all_indices, window = FORECASTING_WINDOW, step_size = FORECASTING_STEP_SIZE)

            instance_daily_input = instance_daily_input.to(device=device,dtype=torch.float32)
            instance_output = instance_output.to(device=device,dtype=torch.float32)

            encoder_x = instance_daily_input.to(device=device, dtype=torch.float32)
        
            decoder_past_x = getPastOutputs(all_outputs, data_indices[data_index], all_indices, window = 45, step_size = FORECASTING_STEP_SIZE)
            decoder_past_x = decoder_past_x.to(device=device, dtype=torch.float32)
            decoder_past_x = ((decoder_past_x - min_swe)/(max_swe - min_swe))

            output_accumulator = []

            for loc_index in range(0, encoder_x.shape[0], batch_size):
                batch_encoder_x = encoder_x[loc_index:loc_index+batch_size,:,:]
                batch_decoder_x = decoder_past_x[loc_index:loc_index+batch_size,:,:]

                for auto_reg_index in range(FORECASTING_WINDOW):
                    tgt_len = batch_decoder_x.shape[1]
                    decoder_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device), diagonal=1).bool()

                    output = swe_model(batch_encoder_x, batch_decoder_x, decoder_attn_mask=decoder_mask)

                    selected_output = output[:, -1:, :]
                    batch_decoder_x = torch.cat((batch_decoder_x, selected_output), dim=1)

                output_accumulator.append(batch_decoder_x[:,-FORECASTING_WINDOW:,:])

            instance_output = instance_output.detach().cpu()
            final_output = torch.cat(output_accumulator,dim = 0).detach().cpu() * max_swe
            
            output_loss = mse_loss(final_output.reshape(-1,FORECASTING_WINDOW), instance_output.reshape(-1,FORECASTING_WINDOW))  

            final_output = final_output.detach().cpu().squeeze().permute(1,0)
            instance_output = instance_output.permute(1,0)

            collected_forecasted_outputs.append(final_output)
            actual_outputs.append(instance_output)

            print('Loss in day :{} , loss : {}'.format(data_index+1, output_loss.item()), final_output.shape)

    all_predicted_outputs = torch.cat(collected_forecasted_outputs,dim=0).reshape(-1,len(all_indices),1).permute(1,0,2).detach().cpu().numpy()
    actual_daily_outputs  = torch.cat(actual_outputs, dim = 0).unsqueeze(2).permute(1,0,2).detach().cpu().numpy() 

    print("Actual Outputs shape", actual_daily_outputs.shape)
    print("Predicted Outputs shape", all_predicted_outputs.shape)

    return actual_daily_outputs, all_predicted_outputs

swe_model.eval()

print("Test Set 1")
daily_window_actual_1, daily_window_predicted_1  = TestModelSP(testing_yr_indices_1, model_inputs, model_outputs, all_dates)
daily_actual_1, daily_predicted_1  = getDailyPredictions(testing_yr_indices_1, model_inputs, model_outputs, all_dates)

print("================================================")

print("Test Set 2")
daily_window_actual_2, daily_window_predicted_2 = TestModelSP(testing_yr_indices_2, model_inputs, model_outputs, all_dates)
daily_actual_2, daily_predicted_2  = getDailyPredictions(testing_yr_indices_2, model_inputs, model_outputs, all_dates)

print("================================================")

print("Test Set 3")
daily_window_actual_3, daily_window_predicted_3 = TestModelSP(testing_yr_indices_3, model_inputs, model_outputs, all_dates)
daily_actual_3, daily_predicted_3  = getDailyPredictions(testing_yr_indices_3, model_inputs, model_outputs, all_dates)

print("================================================")

print("Test Set 4")
daily_window_actual_4, daily_window_predicted_4 = TestModelSP(testing_yr_indices_4, model_inputs, model_outputs, all_dates)
daily_actual_4, daily_predicted_4  = getDailyPredictions(testing_yr_indices_4, model_inputs, model_outputs, all_dates)

print("================================================")

print("Test Set 5")
daily_window_actual_5, daily_window_predicted_5 = TestModelSP(testing_yr_indices_5, model_inputs, model_outputs, all_dates)
daily_actual_5, daily_predicted_5  = getDailyPredictions(testing_yr_indices_5, model_inputs, model_outputs, all_dates)

print("================================================")

joblib.dump(daily_window_predicted_1, '{}/results/predict/2014_daily_w_transformer_predict.pkl'.format(BASE_URL))
joblib.dump(daily_window_predicted_2, '{}/results/predict/2015_daily_w_transformer_predict.pkl'.format(BASE_URL))
joblib.dump(daily_window_predicted_3, '{}/results/predict/2016_daily_w_transformer_predict.pkl'.format(BASE_URL))
joblib.dump(daily_window_predicted_4, '{}/results/predict/2017_daily_w_transformer_predict.pkl'.format(BASE_URL))
joblib.dump(daily_window_predicted_5, '{}/results/predict/2018_daily_w_transformer_predict.pkl'.format(BASE_URL))

joblib.dump(daily_predicted_1, '{}/results/predict/2014_daily_transformer_predict.pkl'.format(BASE_URL))
joblib.dump(daily_predicted_2, '{}/results/predict/2015_daily_transformer_predict.pkl'.format(BASE_URL))
joblib.dump(daily_predicted_3, '{}/results/predict/2016_daily_transformer_predict.pkl'.format(BASE_URL))
joblib.dump(daily_predicted_4, '{}/results/predict/2017_daily_transformer_predict.pkl'.format(BASE_URL))
joblib.dump(daily_predicted_5, '{}/results/predict/2018_daily_transformer_predict.pkl'.format(BASE_URL))

joblib.dump(daily_window_actual_1, '{}/results/actual/2014_daily_w_transformer_actual.pkl'.format(BASE_URL))
joblib.dump(daily_window_actual_2, '{}/results/actual/2015_daily_w_transformer_actual.pkl'.format(BASE_URL))
joblib.dump(daily_window_actual_3, '{}/results/actual/2016_daily_w_transformer_actual.pkl'.format(BASE_URL))
joblib.dump(daily_window_actual_4, '{}/results/actual/2017_daily_w_transformer_actual.pkl'.format(BASE_URL))
joblib.dump(daily_window_actual_5, '{}/results/actual/2018_daily_w_transformer_actual.pkl'.format(BASE_URL))

joblib.dump(daily_actual_1, '{}/results/actual/2014_daily_transformer_actual.pkl'.format(BASE_URL))
joblib.dump(daily_actual_2, '{}/results/actual/2015_daily_transformer_actual.pkl'.format(BASE_URL))
joblib.dump(daily_actual_3, '{}/results/actual/2016_daily_transformer_actual.pkl'.format(BASE_URL))
joblib.dump(daily_actual_4, '{}/results/actual/2017_daily_transformer_actual.pkl'.format(BASE_URL))
joblib.dump(daily_actual_5, '{}/results/actual/2018_daily_transformer_actual.pkl'.format(BASE_URL))
