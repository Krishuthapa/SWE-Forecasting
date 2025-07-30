import os
import sys

import torch
import torch.nn as nn
from torch import optim

import numpy as np
import pandas as pd

import time

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
training_data_indices = dataloader.getTrainingDataIndices()
training_data_indices = np.array(training_data_indices)[:-(FORECASTING_WINDOW * FORECASTING_STEP_SIZE)]

testing_yr_indices_1 , testing_yr_indices_2, testing_yr_indices_3, testing_yr_indices_4, testing_yr_indices_5  = dataloader.getTestingDataIndices()

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
parameters = swe_model.parameters()

# Optimizer
optimizer_sp = torch.optim.AdamW(parameters, lr= 0.001, weight_decay = 0.01)
scheduler_sp = torch.optim.lr_scheduler.StepLR(optimizer_sp, step_size = 2 ,gamma = 0.45, last_epoch= -1, verbose=False)

swe_model.train()

def TrainModelSP(all_inputs, all_outputs, training_yr_indices, epoch_number, batch_size = 32):
    total_loss = 0

    for index, input_index in enumerate(training_yr_indices):
        all_indices = sorted(list(train_loc_indices) + list(test_indices))

        instance_daily_input  = getInputs(all_inputs, input_index, all_indices, short_window = 365, step_size = FORECASTING_STEP_SIZE)
        instance_output = getFutureOutputs(all_outputs, input_index, all_indices, window = FORECASTING_WINDOW, step_size = FORECASTING_STEP_SIZE)

        instance_daily_input = instance_daily_input.to(device=device,dtype=torch.float32)
        instance_output = instance_output.to(device=device,dtype=torch.float32)

        encoder_x = instance_daily_input.to(device=device, dtype=torch.float32)
        
        decoder_past_x = getPastOutputs(all_outputs, input_index, all_indices, window = 45, step_size = FORECASTING_STEP_SIZE)

        decoder_past_x = decoder_past_x.to(device=device, dtype=torch.float32)

        decoder_past_x = ((decoder_past_x - min_swe)/(max_swe - min_swe))

        padding_val =((instance_output - min_swe)/(max_swe - min_swe)).reshape(-1,FORECASTING_WINDOW,1)[:,:-1,:]
        dec_padding_x = padding_val * torch.ones([instance_daily_input.shape[0], FORECASTING_WINDOW-1, decoder_past_x.shape[-1]]).to(dtype=torch.float32, device= device)

        decoder_x = torch.cat((decoder_past_x,dec_padding_x),dim =1)
        decoder_mask = torch.triu(torch.ones(decoder_x.shape[1], decoder_x.shape[1]), diagonal=1).bool().to(device=device)
        
        optimizer_sp.zero_grad()

        output_accumulator = []

        loss = None

        for loc_index in range(0, encoder_x.shape[0], batch_size):
            output = swe_model(encoder_x[loc_index:loc_index+batch_size,:,:], decoder_x[loc_index:loc_index+batch_size,:,:], decoder_attn_mask= decoder_mask)
            output_accumulator.append(output)
            
        final_output = torch.cat(output_accumulator,dim = 0)

        scaled_output = ((instance_output - min_swe)/(max_swe - min_swe)).reshape(-1,FORECASTING_WINDOW).float()
        loss = mse_error(final_output.reshape(-1,FORECASTING_WINDOW).float(), scaled_output)  

        total_loss+=loss.item()

        print('Epoch Number: {} => Batch number :{} , shape: {}, loss value : {} '.format(epoch_number,index+1, encoder_x.shape, loss.item()))
        print("=================================================")

        loss.backward()
        optimizer_sp.step()

        del instance_output
        del decoder_mask
        del final_output
        del decoder_x
        del encoder_x
        del instance_daily_input
        del output_accumulator

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("########################################## Epoch Result Overall ##########################################")
    print('Epoch Number: {} => Avg loss value : {} '.format(epoch_number, total_loss / len(training_yr_indices)))
    print("########################################## Epoch Result Overall ##########################################")
    
    return total_loss / len(training_yr_indices)

hasFile = os.path.isfile("{}/checkpoints/forecasting_daily_tf_model.pt".format(BASE_URL))

strike_count = 0
train_epoch_avg_losses_sp = []

start_time = time.time()
epoch_number = 1

best_nse = float('inf')

scaler = torch.amp.GradScaler(device=device)

if hasFile:
    checkpoint = torch.load("{}/checkpoints/forecasting_daily_tf_model.pt".format(BASE_URL))
            
    if checkpoint['epoch'] < 30:
        epoch_number = checkpoint['epoch']
        strike_count = checkpoint['strike_count']
        swe_model.load_state_dict(checkpoint['swe_model_dict']),
        optimizer_sp.load_state_dict(checkpoint['optimizer_dict']),
        scheduler_sp.load_state_dict(checkpoint['scheduler_state_dict'])

while strike_count <= 3 and epoch_number <= 7:
            
    final_representations = None
    gp_training_outputs = None

    epoch_loss_sp = TrainModelSP(model_inputs,model_outputs,training_data_indices,epoch_number)

    epoch_number += 1
            
    if len(train_epoch_avg_losses_sp) == 0 :
        train_epoch_avg_losses_sp.append(epoch_loss_sp)
        continue
            
    if epoch_loss_sp < best_nse:
        best_nse = epoch_loss_sp
        strike_count = 0

        torch.save({
                'strike_count': strike_count,
                'epoch': epoch_number,
                'swe_model_dict': swe_model.state_dict(),
                'optimizer_dict':optimizer_sp.state_dict(),
                'scheduler_state_dict': scheduler_sp.state_dict(),
                }, "{}/checkpoints/forecasting_daily_tf_model.pt".format(BASE_URL))

    else:
        strike_count += 1
            
    train_epoch_avg_losses_sp.append(epoch_loss_sp)

    scheduler_sp.step()

end_time = time.time()

print("Time Elapsed:", end_time - start_time)
print("Training Epoch Losses:", train_epoch_avg_losses_sp)

