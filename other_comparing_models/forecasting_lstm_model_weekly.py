import torch
import torch.nn as nn

import random

import pandas as pd
import numpy as np

import os
import sys

import math

import time
import joblib

import hydroeval as he
import extended_dl_forecasting as DL

from contextlib import redirect_stdout

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if torch.cuda.is_available():
    print("GPU is running.") 
    dev = "cuda:0" 
else: 
    print("CPU is running.")
    dev = "cpu" 

device = torch.device(dev)

baseUrl = '</path/to/data_and_result_parent_dir>'

# Specify output file paths
output_1 = '{}/outputs/forecasting_lstm_weekly_tr_act_loss.txt'.format(baseUrl)
output_2 = '{}/outputs/forecasting_lstm_weekly_ts_act_loss.txt'.format(baseUrl)

# Config
forecasting_window = 4
forecasting_step_size = 7

snotel_locations_info = pd.read_csv('{}/data/snotel_location_info_2.csv'.format(baseUrl))

test_indices = [43, 66, 338, 463, 115, 105, 83, 306, 481, 21, 185, 136, 110, 406, 408, 132, 
                120, 197, 7, 301, 375, 236, 269, 287, 49, 241, 212, 145, 264, 435, 73, 277, 364, 
                438, 372, 368, 243, 493, 497, 14, 32, 433, 313, 213, 42, 475, 160, 190, 281, 285, 
                316, 461, 154, 424, 310, 340, 179, 167, 193, 466, 56, 158, 365, 492, 432, 415, 68, 
                409, 76, 148, 267, 227, 91, 189, 472, 162, 147, 395, 113, 2, 271, 357, 347, 335, 507, 
                465, 57, 19, 442, 381, 398, 79, 337, 501, 41, 339, 331, 63, 359, 232, 323, 258]

test_indices = sorted(test_indices)

# Data loader to call all the useful data.
dataloader = DL.DataLoader(test_indices = test_indices, temp_window = 7)
all_locations_distances, all_locations_angularities = dataloader.getDistanceAndAngularity()

dataloader.cleanDataJunk()
dataloader.normalizeData()
dataloader.addDayInfo()
dataloader.prepareInputsAndOutputs()

model_inputs, model_outputs = dataloader.getModelInputsAndOutputs()
dataloader.completeInputsAndOutputs()

training_data_indices = dataloader.getTrainingDataIndices()

# Train Test indices collection.
training_data_indices = np.array(training_data_indices)[:-(forecasting_window * forecasting_step_size)]
testing_yr_indices_1 , testing_yr_indices_2, testing_yr_indices_3, testing_yr_indices_4, testing_yr_indices_5  = dataloader.getTestingDataIndices()
train_loc_indices = dataloader.getTrainingIndices()

# Gathering the required prompts for the model.
train_prompts, test_prompts, all_prompts = dataloader.getModelPrompts()
train_prompts = torch.from_numpy(train_prompts).to(device=device, dtype= torch.float32)
test_prompts = torch.from_numpy(test_prompts).to(device=device, dtype= torch.float32)
all_prompts = torch.from_numpy(all_prompts).to(device=device, dtype = torch.float32)

# Model Inputs and Outputs.
model_inputs = torch.from_numpy(model_inputs).to(device=device, dtype= torch.float32)
model_outputs = torch.from_numpy(model_outputs).to(device=device, dtype= torch.float32)

max_swe,min_swe = dataloader.getMaxMinSWE()

# Encoder Model
class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(EncoderLSTM, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell  # Return the final hidden and cell states

# Decoder Model
class DecoderLSTM(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers):
        super(DecoderLSTM, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = self.fc(output)
        return output, hidden, cell

# Encoder-Decoder Model
class SweLSTM(nn.Module):
    def __init__(self, input_dim, sp_dim, prompt_dim, hidden_dim, output_dim, num_layers, forecasting_length):
        super(SweLSTM, self).__init__()

        self.spatial_embedding_1 = nn.Linear(sp_dim, int(hidden_dim/2))
        self.spatial_embedding_2 = nn.Linear(int(hidden_dim/2), int(hidden_dim))

        self.prompt_embedding_1 = nn.Linear(prompt_dim, int(hidden_dim/2))
        self.prompt_embedding_2 = nn.Linear(int(hidden_dim/2), hidden_dim)

        self.dropout_layer = nn.Dropout(p=0.05)
        self.activation_function = nn.GELU()

        self.encoder_daily = EncoderLSTM(input_dim, hidden_dim, num_layers)
        self.encoder_weekly = EncoderLSTM(input_dim, hidden_dim, num_layers)
        self.encoder_yearly = EncoderLSTM(input_dim, hidden_dim, num_layers)

        self.decoder = DecoderLSTM(output_dim, 3 * hidden_dim, num_layers)

        self.forecasting_length = forecasting_length

    def forward(self, x_daily, x_weekly, x_yearly, prompts):
        # Encode the input sequence
        batch_size, days_count, inp_size = x_daily.shape
        batch_size, weeks_count, inp_size = x_weekly.shape
        batch_size, years_count , inp_size = x_yearly.shape

        sp_tp = x_daily[:, -1, 0:4].reshape(-1,4)

        sp_embedding = self.spatial_embedding_1(sp_tp)
        sp_embedding = self.dropout_layer(sp_embedding)
        sp_embedding = self.activation_function(sp_embedding)

        sp_embedding = self.spatial_embedding_2(sp_embedding)

        prompt_embedding = self.prompt_embedding_1(prompts)
        prompt_embedding = self.dropout_layer(prompt_embedding)
        prompt_embedding = self.activation_function(prompt_embedding)

        prompt_embedding = self.prompt_embedding_2(prompt_embedding)

        location_embedding = sp_embedding + prompt_embedding
        location_embedding = location_embedding.reshape(1, batch_size, -1)

        source_daily = x_daily[:,:,4:]
        hidden_daily, cell_daily = self.encoder_daily(source_daily)

        source_weekly = x_weekly[:,:,4:]
        hidden_weekly, cell_weekly = self.encoder_weekly(source_daily)

        source_yearly = x_yearly[:,:,4:]
        hidden_yearly, cell_yearly = self.encoder_weekly(source_yearly)

        hidden_daily = hidden_daily  + location_embedding
        cell_daily = cell_daily  + location_embedding

        hidden_weekly = hidden_weekly + location_embedding
        cell_weekly = cell_weekly + location_embedding

        hidden_yearly = hidden_yearly + location_embedding
        cell_yearly = cell_yearly + location_embedding

        cell = torch.cat((cell_daily,cell_weekly, cell_yearly), dim = -1)
        hidden = torch.cat((hidden_daily,hidden_weekly, hidden_yearly), dim = -1)

        target = torch.ones(batch_size,self.forecasting_length,1).to(device=device, dtype=torch.float32)
        target = target * -1

        # Decode the target sequence
        outputs = []
        decoder_input = target[:, 0, :].unsqueeze(1)

        for t in range(self.forecasting_length):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(output)
            
            decoder_input = output

        final_outputs = torch.cat(outputs, dim=1)

        return final_outputs

swe_model = SweLSTM(input_dim = 15, sp_dim = 4, prompt_dim = 1536, hidden_dim = 512, output_dim = 1, num_layers = 1, forecasting_length = forecasting_window)
swe_model = swe_model.to(device=device)

# Loss Function 
mae_error = nn.MSELoss()
parameters = swe_model.parameters()

# Optimizer
optimizer_sp = torch.optim.AdamW(parameters, lr= 0.0005, weight_decay = 0.0001)
scheduler_sp = torch.optim.lr_scheduler.StepLR(optimizer_sp, step_size = 2 ,gamma = 0.65, last_epoch= -1, verbose=False)

swe_model.train()

original_stdout = sys.stdout

def getInputs(index,loc_indices = train_loc_indices, window = 15, step_size = 1):
    indices = sorted([index - (value * step_size) for value in range(0, window)])

    inputs = model_inputs[:,loc_indices,:][indices,:,:].reshape(window, len(loc_indices), -1)

    return inputs.permute(1,0,2)

def getOutputs(index, loc_indices = train_loc_indices, window = 15, step_size = 1):
    indices = sorted([index + (value * step_size) for value in range(1, window+1)])

    outputs = model_outputs[:,loc_indices,:][indices,:,:].permute(1,0,2)

    return outputs.reshape(len(loc_indices), len(indices))

def TrainModelSP(training_yr_indices, loc_prompts, location_distances, location_angularities, epoch_number):
    total_loss = 0
    total_batches = 0

    for index, input_index in enumerate(training_yr_indices):
        all_indices = sorted(list(train_loc_indices) + list(test_indices))

        instance_daily_input  = getInputs(input_index, all_indices, window = 30, step_size = 1)
        instance_weekly_input = getInputs(input_index, all_indices, window = 30, step_size = 7)
        instance_yearly_input = getInputs(input_index, all_indices, window = 5, step_size = 365)

        instance_output = getOutputs(input_index, all_indices, window = forecasting_window, step_size = forecasting_step_size)

        instance_daily_input = torch.tensor(instance_daily_input).to(device=device,dtype=torch.float32)
        instance_weekly_input = torch.tensor(instance_weekly_input).to(device = device, dtype= torch.float32)
        instance_yearly_input = torch.tensor(instance_yearly_input).to(device=device, dtype= torch.float32)

        instance_output = torch.tensor(instance_output).to(device=device,dtype=torch.float32)
        loc_prompts = torch.tensor(loc_prompts).to(device=device,dtype=torch.float32)

        output = swe_model(instance_daily_input, instance_weekly_input, instance_yearly_input, loc_prompts)
            
        loss = mae_error(output.reshape(instance_daily_input.shape[0], forecasting_window), instance_output)
        
        total_loss+=loss.item()
        
        optimizer_sp.zero_grad()
        loss.backward()

        optimizer_sp.step()
        
        if  (index + 1) % 100 == 0:
            print('Epoch Number: {} => Batch number :{} , loss value : {} '.format(epoch_number,index+1,loss.item()))
            print('Length:{}'.format(instance_daily_input.shape[1]))
            print("=================================================")
        
    print('Epoch Number: {} => Avg loss value : {} '.format(epoch_number, total_loss / len(training_yr_indices)))
    
    return total_loss / len(training_yr_indices)

hasFile = os.path.isfile("{}/checkpoints/forecasting_weekly_lstm_model.pt".format(baseUrl))

with open(output_1, 'w') as file1:
    with redirect_stdout(file1):
        
        strike_count = 0
        train_epoch_avg_losses_sp = []

        start_time = time.time()
        epoch_number = 1

        best_nse = float('inf')

        if hasFile:
            checkpoint = torch.load("{}/checkpoints/forecasting_weekly_lstm_model.pt".format(baseUrl))
            
            if checkpoint['epoch'] < 30:
                epoch_number = checkpoint['epoch']
                strike_count = checkpoint['strike_count']
                swe_model.load_state_dict(checkpoint['swe_model_dict']),
                optimizer_sp.load_state_dict(checkpoint['optimizer_dict']),

        while strike_count <= 3 and epoch_number <= 10:
            
            final_representations = None
            gp_training_outputs = None

            epoch_loss_sp = TrainModelSP(training_data_indices, all_prompts, all_locations_distances, all_locations_angularities,epoch_number)

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
                }, "{}/checkpoints/forecasting_weekly_lstm_model.pt".format(baseUrl))

            else:
                strike_count += 1
    
            train_epoch_avg_losses_sp.append(epoch_loss_sp)
        
            scheduler_sp.step()

        end_time = time.time()

        print("Time Elapsed:", end_time - start_time)
        print("Training Epoch Losses:", train_epoch_avg_losses_sp)

swe_model.eval()

hasFile = os.path.isfile("{}/checkpoints/forecasting_weekly_lstm_model.pt".format(baseUrl))

if hasFile:
    checkpoint = torch.load("{}/checkpoints/forecasting_weekly_lstm_model.pt".format(baseUrl))
            
    epoch_number = checkpoint['epoch']
    strike_count = checkpoint['strike_count']
    swe_model.load_state_dict(checkpoint['swe_model_dict']),
    optimizer_sp.load_state_dict(checkpoint['optimizer_dict']),

testing_yr_indices_1 , testing_yr_indices_2, testing_yr_indices_3, testing_yr_indices_4, testing_yr_indices_5  = dataloader.getTestingDataIndices()
all_test_indices = list(testing_yr_indices_1) + list(testing_yr_indices_2) +list(testing_yr_indices_3) + list(testing_yr_indices_4) + list(testing_yr_indices_5)

def TestModelSP(testing_yr_indices):
    loss_value = 0
    losses = []
    
    mae_error = nn.MSELoss()
    
    actual_outputs = []
    predicted_outputs = []

    all_indices = sorted(list(train_loc_indices) + list(test_indices))

    with torch.no_grad():
        for index , input_index in enumerate(testing_yr_indices): 

            instance_daily_input  = getInputs(input_index, all_indices, window = 30, step_size = 1)
            instance_weekly_input = getInputs(input_index, all_indices, window = 30, step_size = 7)
            instance_yearly_input = getInputs(input_index, all_indices, window = 5, step_size = 365)

            instance_output = getOutputs(input_index, all_indices, window = forecasting_window, step_size = forecasting_step_size)

            instance_daily_input = torch.tensor(instance_daily_input).to(device=device,dtype=torch.float32)
            instance_weekly_input = torch.tensor(instance_weekly_input).to(device=device,dtype=torch.float32)
            instance_yearly_input = torch.tensor(instance_yearly_input).to(device=device, dtype= torch.float32)

            output = swe_model(instance_daily_input, instance_weekly_input, instance_yearly_input, all_prompts)

            instance_output = instance_output.detach().cpu()
            output = output.detach().cpu()

            output_loss = mae_error(output.reshape(-1,forecasting_window), instance_output.reshape(-1,forecasting_window))            

            actual_outputs.append(instance_output.reshape(1,-1,forecasting_window).cpu().detach().numpy())
            predicted_outputs.append(output.reshape(1,-1,forecasting_window).cpu().numpy())
            
            print('Batch number :{} , loss : {}'.format(index+1, output_loss.item()))

    actual_outputs = np.array(actual_outputs).reshape(-1,len(all_indices),forecasting_window)
    predicted_outputs = np.array(predicted_outputs).reshape(-1,len(all_indices),forecasting_window)
    
    return (actual_outputs, predicted_outputs)

def getDailyPredictions(data_indices):
    actual_outputs = []
    collected_forecasted_outputs = []

    loss = nn.MSELoss()

    all_indices = sorted(list(train_loc_indices) + list(test_indices))

    for data_index in range(0,len(data_indices), forecasting_window * forecasting_step_size):        
        instance_daily_input  = getInputs(data_indices[data_index], all_indices, window = 30, step_size = 1)
        instance_weekly_input  = getInputs(data_indices[data_index], all_indices, window = 30, step_size = 7)
        instance_yearly_input = getInputs(data_indices[data_index], all_indices, window = 5, step_size = 365)

        instance_output = getOutputs(data_indices[data_index], all_indices, window = forecasting_window, step_size = forecasting_step_size)

        actual_outputs.append(instance_output.permute(1,0).reshape(forecasting_window, len(all_indices),1))

        instance_daily_input = torch.tensor(instance_daily_input).to(device=device,dtype=torch.float32)
        instance_weekly_input = torch.tensor(instance_weekly_input).to(device=device,dtype=torch.float32)
        instance_yearly_input = torch.tensor(instance_yearly_input).to(device=device, dtype= torch.float32)
                
        output = swe_model(instance_daily_input, instance_weekly_input, instance_yearly_input,  all_prompts)
        output_loss = mae_error(output.reshape(-1,forecasting_window), instance_output.reshape(-1,forecasting_window))   

        output = output.detach().cpu().reshape(-1,forecasting_window).permute(1,0).reshape(forecasting_window,len(all_indices),1)
        collected_forecasted_outputs.append(output)

        print('Loss in day :{} , loss : {}'.format(data_index+1, output_loss.item()))
    
    all_predicted_outputs = torch.cat(collected_forecasted_outputs,dim=0).permute(1,0,2).detach().cpu().numpy()
    actual_weekly_outputs  = torch.cat(actual_outputs, dim = 0).permute(1,0,2).detach().cpu().numpy() 

    return actual_weekly_outputs, all_predicted_outputs

with open(output_2, 'w') as file1:
    with redirect_stdout(file1):
        print("Test Set 1")
        weekly_window_actual_1, weekly_window_predicted_1  = TestModelSP(testing_yr_indices_1)
        weekly_actual_1, weekly_predicted_1 = getDailyPredictions(testing_yr_indices_1)
        print("================================================")

        print("Test Set 2")
        weekly_window_actual_2, weekly_window_predicted_2 = TestModelSP(testing_yr_indices_2)
        weekly_actual_2, weekly_predicted_2 = getDailyPredictions(testing_yr_indices_2)
        print("================================================")
        
        print("Test Set 3")
        weekly_window_actual_3, weekly_window_predicted_3 = TestModelSP(testing_yr_indices_3)
        weekly_actual_3, weekly_predicted_3 = getDailyPredictions(testing_yr_indices_3)
        print("================================================")
        
        print("Test Set 4")
        weekly_window_actual_4, weekly_window_predicted_4 = TestModelSP(testing_yr_indices_4)
        weekly_actual_4, weekly_predicted_4 = getDailyPredictions(testing_yr_indices_4)
        print("================================================")
        
        print("Test Set 5")
        weekly_window_actual_5, weekly_window_predicted_5 = TestModelSP(testing_yr_indices_5)
        weekly_actual_5, weekly_predicted_5 = getDailyPredictions(testing_yr_indices_5)
        print("================================================")

sys.stdout = original_stdout

joblib.dump(weekly_window_predicted_1, '{}/results/predict/2014_weekly_window_predict_lstm_2.pkl'.format(baseUrl))
joblib.dump(weekly_window_predicted_2, '{}/results/predict/2015_weekly_window_predict_lstm_2.pkl'.format(baseUrl))
joblib.dump(weekly_window_predicted_3, '{}/results/predict/2016_weekly_window_predict_lstm_2.pkl'.format(baseUrl))
joblib.dump(weekly_window_predicted_4, '{}/results/predict/2017_weekly_window_predict_lstm_2.pkl'.format(baseUrl))
joblib.dump(weekly_window_predicted_5, '{}/results/predict/2018_weekly_window_predict_lstm_2.pkl'.format(baseUrl))

joblib.dump(weekly_window_actual_1, '{}/results/actual/2014_weekly_window_actual_lstm.pkl'.format(baseUrl))
joblib.dump(weekly_window_actual_2, '{}/results/actual/2015_weekly_window_actual_lstm.pkl'.format(baseUrl))
joblib.dump(weekly_window_actual_3, '{}/results/actual/2016_weekly_window_actual_lstm.pkl'.format(baseUrl))
joblib.dump(weekly_window_actual_4, '{}/results/actual/2017_weekly_window_actual_lstm.pkl'.format(baseUrl))
joblib.dump(weekly_window_actual_5, '{}/results/actual/2018_weekly_window_actual_lstm.pkl'.format(baseUrl))

joblib.dump(weekly_predicted_1, '{}/results/predict/2014_weekly_predict_lstm_2.pkl'.format(baseUrl))
joblib.dump(weekly_predicted_2, '{}/results/predict/2015_weekly_predict_lstm_2.pkl'.format(baseUrl))
joblib.dump(weekly_predicted_3, '{}/results/predict/2016_weekly_predict_lstm_2.pkl'.format(baseUrl))
joblib.dump(weekly_predicted_4, '{}/results/predict/2017_weekly_predict_lstm_2.pkl'.format(baseUrl))
joblib.dump(weekly_predicted_5, '{}/results/predict/2018_weekly_predict_lstm_2.pkl'.format(baseUrl))

joblib.dump(weekly_actual_1, '{}/results/actual/2014_weekly_actual_lstm.pkl'.format(baseUrl))
joblib.dump(weekly_actual_2, '{}/results/actual/2015_weekly_actual_lstm.pkl'.format(baseUrl))
joblib.dump(weekly_actual_3, '{}/results/actual/2016_weekly_actual_lstm.pkl'.format(baseUrl))
joblib.dump(weekly_actual_4, '{}/results/actual/2017_weekly_actual_lstm.pkl'.format(baseUrl))
joblib.dump(weekly_actual_5, '{}/results/actual/2018_weekly_actual_lstm.pkl'.format(baseUrl))


