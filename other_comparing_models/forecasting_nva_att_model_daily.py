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

from custom_encoder_model import EncoderLayer as CustomEncoderLayer

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
output_1 = '{}/outputs/forecasting_nva_daily_att_tr_act_loss.txt'.format(baseUrl)
output_2 = '{}/outputs/forecasting_nva_daily_att_ts_act_loss.txt'.format(baseUrl)

# Config
forecasting_window = 10
forecasting_step_size = 1

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

class CrossVarTransformer(nn.Module):
    def __init__(self, encoder_input_dim = 13 , batch_size = 7, sp_tp_dim = 6,  model_dim = 512, prompt_dim = 1536, enc_layers_count = 8, nhead = 16):
        super().__init__()
        
        # Storing the passed argument on the class definition.        
        self.model_dim = model_dim
        self.prompt_dim = prompt_dim

        self.encoder_input_dim = encoder_input_dim

        self.batch_size = batch_size
        self.sp_tp_dim = sp_tp_dim
        
        self.nhead = nhead
        self.enc_layers_count = enc_layers_count
        
        self.token_embeds = nn.ModuleList(
                [torch.nn.Linear(self.batch_size, self.model_dim) for i in range(self.encoder_input_dim)])
        
        self.sp_tp_prompt_embed = nn.ModuleList()
        self.sp_tp_prompt_embed.append(torch.nn.Linear(self.sp_tp_dim, int(self.model_dim/2)))
        self.sp_tp_prompt_embed.append(nn.GELU())
        self.sp_tp_prompt_embed.append(nn.Dropout(p = 0.05))
        self.sp_tp_prompt_embed.append(torch.nn.Linear(int(self.model_dim/2), int(self.model_dim)))
        self.sp_tp_prompt_embed = nn.Sequential(*self.sp_tp_prompt_embed)

        self.prompt_embedding = nn.Linear(self.prompt_dim , self.model_dim)

    def getVariableEmbeddingInitialized(self,embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return torch.from_numpy(emb).to(device=device, dtype= torch.float32)

    def forward(self,encoder_inputs, prompt_inputs, distance_pairs, angularity_pairs):
        # Getting the configuration of the passed data.
        inp_batch_size, sequence_length , input_dim = encoder_inputs.shape

        encoder_inputs_historical = encoder_inputs[:, :, 6:]
        encoder_inputs_var_temp = encoder_inputs_historical.permute(1,0,2).reshape(sequence_length, inp_batch_size, -1).permute(0,2,1)

        inputs_sp_tp = encoder_inputs[-1, :, :6].reshape(sequence_length,-1) 

        sp_tp_embedding = self.sp_tp_prompt_embed(inputs_sp_tp)
        sp_tp_embedding = sp_tp_embedding.reshape(1,sequence_length, int(self.model_dim)).permute(1,0,2)

        prompt_embedding_x = self.prompt_embedding(prompt_inputs.reshape(-1,self.prompt_dim))
        prompt_embedding_x = prompt_embedding_x.reshape(1,sequence_length, self.model_dim).permute(1,0,2)

        # Input Embedding Day ==============================================================

        embeds = []
        for index in range(self.encoder_input_dim):
            embeds.append(self.token_embeds[index](encoder_inputs_var_temp[:, index,:].reshape(-1,self.batch_size)))

        embedded_var_input_x = torch.stack(embeds, dim = 0).to(device= device, dtype= torch.float32)
        embedded_var_input_x = embedded_var_input_x.permute(1,0,2)

        # variable_embedding = self.variable_embeddings.repeat(sequence_length,1,1)

        # embedded_var_input_x = embedded_var_input_x + variable_embedding
        query_embedding = sp_tp_embedding + prompt_embedding_x

        all_variable_embedding = embedded_var_input_x.mean(dim = 1)
        all_variable_embedding = all_variable_embedding.reshape(1,-1,self.model_dim)

        input_embedded_features = all_variable_embedding + query_embedding.permute(1,0,2)
        
        return input_embedded_features.reshape(1,sequence_length, self.model_dim)
    
class SWETransformer(nn.Module):
    def __init__(self, model_dim = 1024, sp_tp_dim = 6, prompt_dim = 1536, n_output_heads = 1, enc_layers_count = 8, nhead = 16):
        super().__init__()
        
        # Storing the passed argument on the class definition.        
        self.model_dim = model_dim
        self.sp_tp_dim = sp_tp_dim
        self.prompt_dim = prompt_dim

        self.n_output_heads = n_output_heads
        
        self.nhead = nhead
        self.enc_layers_count = enc_layers_count

        # Historical Inputs Attention
        self.historical_daily_att = CrossVarTransformer(batch_size = 1, model_dim = self.model_dim , prompt_dim = self.prompt_dim)
        self.historical_monthly_att = CrossVarTransformer(batch_size = 120, model_dim = self.model_dim, prompt_dim = self.prompt_dim)
        self.historical_yearly_att = CrossVarTransformer(batch_size = 5, model_dim = self.model_dim, prompt_dim = self.prompt_dim)

        self.encoder_layers = nn.ModuleList([CustomEncoderLayer(d_model = 3 * self.model_dim, num_heads= self.nhead) for _ in range(self.enc_layers_count)])
        
        # Final output layer for the model.
        self.output_layers = []
        self.output_layers.append(torch.nn.Linear(self.model_dim * 3, self.model_dim))
        self.output_layers.append(nn.GELU())
        self.output_layers.append(nn.Dropout(p =0.05))
        self.output_layers.append(torch.nn.Linear(self.model_dim, 128))
        self.output_layers.append(nn.GELU())
        self.output_layers.append(torch.nn.Linear(128, 8))
        self.output_layers.append(nn.GELU())
        self.output_layers = nn.Sequential(*self.output_layers)

        self.final_output = torch.nn.Linear(8, self.n_output_heads)
                 
    def forward(self, historical_daily_inputs, historical_monthly_inputs, historical_yearly_inputs, loc_prompt_inputs, token_distances, token_angularities):  

        # Historical Attention Info
        historical_daily_encoded = self.historical_daily_att(historical_daily_inputs, loc_prompt_inputs, token_distances, token_angularities)
        historical_monthly_encoded = self.historical_monthly_att(historical_monthly_inputs, loc_prompt_inputs,token_distances, token_angularities)
        historical_yearly_encoded = self.historical_yearly_att(historical_yearly_inputs, loc_prompt_inputs, token_distances, token_angularities)

        merged_input_x = torch.cat((historical_daily_encoded, historical_monthly_encoded, historical_yearly_encoded), dim = 2)
        final_output = merged_input_x

        for encoder_layer in self.encoder_layers:
            final_output = encoder_layer(final_output, token_distances, token_angularities)
        
        final_output = final_output + merged_input_x
        
        final_representation = self.output_layers(final_output.reshape(-1, 3 * self.model_dim))

        outputs = self.final_output(final_representation)
        outputs = outputs.reshape(1 , historical_daily_inputs.shape[1] , self.n_output_heads)

        final_representation = final_representation.reshape(1, historical_daily_inputs.shape[1], 8)
        
        return outputs, final_representation

swe_model = SWETransformer(model_dim = 1024, prompt_dim = 1536, n_output_heads = forecasting_window, nhead = 16)
swe_model = swe_model.to(device=device)

# Loss Function 
mae_error = nn.MSELoss()
parameters = swe_model.parameters()

# Optimizer
optimizer_sp = torch.optim.AdamW(parameters, lr= 0.00005, weight_decay = 0.0001)
scheduler_sp = torch.optim.lr_scheduler.StepLR(optimizer_sp, step_size = 2 ,gamma = 0.5, last_epoch= -1, verbose=False)

def getRandomizedInputOutput(daily_inputs, monthly_inputs, yearly_inputs, prompt_inputs, all_distances, all_angularities, outputs):

    seq_length = daily_inputs.shape[1]
    
    # shuffle sequence
    indices = np.arange(seq_length)
    np.random.shuffle(indices)

    shuffled_daily_input = daily_inputs[:,indices,:]
    shuffled_monthly_input = monthly_inputs[:,indices,:]
    shuffled_yearly_input = yearly_inputs[:,indices,:]
    shuffled_prompt_input = prompt_inputs[:,indices,:]

    shuffled_distances = all_distances[indices,:][:,indices]
    shuffled_angularities = all_angularities[indices,:][:,indices]
    
    shuffled_output = outputs[indices,:]

    sampled_daily_inputs = []
    sampled_monthly_inputs = []
    sampled_yearly_inputs = []
    sampled_prompt_inputs = []

    sampled_distances = []
    sampled_angularities = []

    sampled_outputs = []
    
    for sample in range(2):
        # sample sequence
        random_length = random.randint(50,seq_length)
        
        chosen_indices = random.sample([value for value in range(seq_length)],random_length)
    
        sampled_daily_input = shuffled_daily_input[:,chosen_indices,:]
        sampled_monthly_input = shuffled_monthly_input[:,chosen_indices,:]
        sampled_yearly_input = shuffled_yearly_input[:,chosen_indices,:]
        sampled_prompt_input = shuffled_prompt_input[:,chosen_indices,:]

        sampled_distance = shuffled_distances[chosen_indices,:][:,chosen_indices]
        sampled_angularity = shuffled_angularities[chosen_indices, :][:, chosen_indices]

        sampled_output = shuffled_output[chosen_indices,:]

        sampled_daily_inputs.append(sampled_daily_input)
        sampled_monthly_inputs.append(sampled_monthly_input)
        sampled_yearly_inputs.append(sampled_yearly_input)
        sampled_prompt_inputs.append(sampled_prompt_input)

        sampled_distances.append(sampled_distance)
        sampled_angularities.append(sampled_angularity)

        sampled_outputs.append(sampled_output)
        
    return sampled_daily_inputs, sampled_monthly_inputs, sampled_yearly_inputs, sampled_prompt_inputs, sampled_distances, sampled_angularities, sampled_outputs

swe_model.train()

original_stdout = sys.stdout

def getInputs(index,loc_indices = train_loc_indices, window = 15, step_size = 1):
    indices = sorted([index - (value * step_size) for value in range(0, window)])

    inputs = model_inputs[:,loc_indices,:][indices,:,:].reshape(window, len(loc_indices), -1)

    return inputs

def getOutputs(index, loc_indices = train_loc_indices, window = 15, step_size = 1):
    indices = sorted([index + (value * step_size) for value in range(1, window+1)])

    outputs = model_outputs[:,loc_indices,:][indices,:,:].permute(1,0,2).reshape(len(loc_indices),-1)

    return outputs

def TrainModelSP(training_yr_indices, loc_prompts, location_distances, location_angularities, epoch_number):
    total_loss = 0
    total_batches = 0

    for index, input_index in enumerate(training_yr_indices):
        all_indices = sorted(list(train_loc_indices) + list(test_indices))

        instance_daily_input  = getInputs(input_index, all_indices, window = 1, step_size = 1)
        instance_monthly_input = getInputs(input_index, all_indices, window = 120, step_size = 7)
        instance_yearly_input  = getInputs(input_index, all_indices, window = 5, step_size = 365)

        instance_output = getOutputs(input_index, all_indices, window = forecasting_window, step_size = forecasting_step_size)
                    
        sampled_daily_input = torch.tensor(instance_daily_input).to(device=device,dtype=torch.float32)
        sampled_monthly_input = torch.tensor(instance_monthly_input).to(device=device,dtype=torch.float32)
        sampled_yearly_input = torch.tensor(instance_yearly_input).to(device=device,dtype=torch.float32)

        sampled_prompt_input = all_prompts
        sampled_output = torch.tensor(instance_output).to(device=device,dtype=torch.float32)

        sampled_distance = torch.tensor(location_distances).to(device=device, dtype=torch.float32)
        sampled_angularity = torch.tensor(location_angularities).to(device=device, dtype=torch.float32)

        output, final_representation = swe_model(sampled_daily_input, sampled_monthly_input, sampled_yearly_input, sampled_prompt_input, sampled_distance, sampled_angularity)
            
        loss = mae_error(output.reshape(sampled_daily_input.shape[1], -1), sampled_output)
        
        total_loss+=loss.item()
        
        optimizer_sp.zero_grad()
        loss.backward()

        optimizer_sp.step()
        
        if  (index + 1) % 100 == 0:
            print('Epoch Number: {} => Batch number :{} , loss value : {} '.format(epoch_number,index+1,loss.item()))
            print('Length:{}'.format(sampled_daily_input.shape[1]))
            print("=================================================")
        
    print('Epoch Number: {} => Avg loss value : {} '.format(epoch_number, total_loss / len(training_yr_indices)))
    
    return total_loss / len(training_yr_indices)

hasFile = os.path.isfile("{}/checkpoints/forecasting_nva_att_model_daily.pt".format(baseUrl))

with open(output_1, 'w') as file1:
    with redirect_stdout(file1):
        
        strike_count = 0
        train_epoch_avg_losses_sp = []

        start_time = time.time()
        epoch_number = 1

        best_nse = float('inf')

        if hasFile:
            checkpoint = torch.load("{}/checkpoints/forecasting_nva_att_model_daily.pt".format(baseUrl))
            
            if checkpoint['epoch'] < 30:
                epoch_number = checkpoint['epoch']
                strike_count = checkpoint['strike_count']
                swe_model.load_state_dict(checkpoint['swe_model_dict']),
                optimizer_sp.load_state_dict(checkpoint['optimizer_dict']),

        while strike_count <= 3 and epoch_number <= 10:
            
            final_representations = None
            gp_training_outputs = None

            epoch_loss_sp = TrainModelSP(training_data_indices, train_prompts, all_locations_distances, all_locations_angularities,epoch_number)

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
                }, "{}/checkpoints/forecasting_nva_att_model_daily.pt".format(baseUrl))

            else:
                strike_count += 1
    
            train_epoch_avg_losses_sp.append(epoch_loss_sp)
        
            #scheduler_sp.step()

        end_time = time.time()

        print("Time Elapsed:", end_time - start_time)
        print("Training Epoch Losses:", train_epoch_avg_losses_sp)

swe_model.eval()

hasFile = os.path.isfile("{}/checkpoints/forecasting_nva_att_model_daily.pt".format(baseUrl))

if hasFile:
    checkpoint = torch.load("{}/checkpoints/forecasting_nva_att_model_daily.pt".format(baseUrl))
            
    epoch_number = checkpoint['epoch']
    strike_count = checkpoint['strike_count']
    swe_model.load_state_dict(checkpoint['swe_model_dict']),
    optimizer_sp.load_state_dict(checkpoint['optimizer_dict']),

def getFinalInputsForGP(training_yr_indices, train_prompts):
    total_loss = 0
    total_batches = 0

    input_learnt_representations = []
    predicted_outputs = []
    actual_outputs = []
    
    with torch.no_grad():
        for index, input_index in enumerate(training_yr_indices):
            all_indices = sorted(list(train_loc_indices) + list(test_indices))
            
            instance_daily_input  = getInputs(input_index, all_indices, window = 1, step_size = 1)
            instance_monthly_input = getInputs(input_index, all_indices, window = 120, step_size = 1)
            instance_yearly_input  = getInputs(input_index, all_indices, window = 5, step_size = 365)

            instance_output = getOutputs(input_index, all_indices, window = forecasting_window, step_size = forecasting_step_size)

            instance_distances = torch.from_numpy(all_locations_distances[all_indices,:][:,all_indices]).to(device=device,dtype=torch.float32)
            instance_angularities = torch.from_numpy(all_locations_angularities[all_indices,:][:,all_indices]).to(device=device, dtype=torch.float32)
            
            instance_daily_input = torch.tensor(instance_daily_input).to(device=device,dtype=torch.float32)
            instance_monthly_input = torch.tensor(instance_monthly_input).to(device=device,dtype=torch.float32)
            instance_yearly_input = torch.tensor(instance_yearly_input).to(device=device,dtype=torch.float32)

            output, final_representation = swe_model(instance_daily_input, instance_monthly_input, instance_yearly_input, all_prompts, instance_distances, instance_angularities)

            final_representation = final_representation.detach().cpu()
            instance_output = instance_output.detach().cpu()
            output = output.detach().cpu()
            
            input_learnt_representations.append(final_representation)
            predicted_outputs.append(output.reshape(1,-1,10))
            actual_outputs.append(instance_output.reshape(1,-1,10))

    torch.save(input_learnt_representations, '{}/outputs/forecasting/forecasting_nva_daily_att_inputs_for_gp_act.pt'.format(baseUrl))
    torch.save(predicted_outputs, '{}/outputs/forecasting/forecasting_nva_daily_att_predicted_outputs_for_gp_act.pt'.format(baseUrl))
    torch.save(actual_outputs, '{}/outputs/forecasting/forecasting_nva_daily_att_outputs_for_gp_act.pt'.format(baseUrl))

getFinalInputsForGP(training_data_indices, train_prompts)

testing_yr_indices_1 , testing_yr_indices_2, testing_yr_indices_3, testing_yr_indices_4, testing_yr_indices_5  = dataloader.getTestingDataIndices()
all_test_indices = list(testing_yr_indices_1) + list(testing_yr_indices_2) +list(testing_yr_indices_3) + list(testing_yr_indices_4) + list(testing_yr_indices_5)

def TestModelSP(testing_yr_indices):
    loss_value = 0
    losses = []
    
    mae_error = nn.MSELoss()
    
    actual_outputs = []
    predicted_outputs = []
    final_representations = []

    all_indices = sorted(list(train_loc_indices) + list(test_indices))

    with torch.no_grad():
        for index , input_index in enumerate(testing_yr_indices): 

            comparing_inputs = model_inputs[:,train_loc_indices,:][input_index,:,:]
            current_instance_input = model_inputs[:,all_indices,:][input_index,:,:]

            instance_daily_input  = getInputs(input_index, all_indices, window = 1, step_size = 1)
            instance_monthly_input = getInputs(input_index, all_indices, window = 120, step_size = 1)
            instance_yearly_input  = getInputs(input_index, all_indices, window = 5, step_size = 365)

            instance_output = getOutputs(input_index, all_indices, window = forecasting_window, step_size = forecasting_step_size)

            instance_distances = torch.from_numpy(all_locations_distances[all_indices,:][:,all_indices]).to(device=device,dtype=torch.float32)
            instance_angularities = torch.from_numpy(all_locations_angularities[all_indices,:][:,all_indices]).to(device=device, dtype=torch.float32)

            instance_daily_input = torch.tensor(instance_daily_input).to(device=device,dtype=torch.float32)
            instance_monthly_input = torch.tensor(instance_monthly_input).to(device=device,dtype=torch.float32)
            instance_yearly_input = torch.tensor(instance_yearly_input).to(device=device,dtype=torch.float32)
                        
            output, final_representation = swe_model(instance_daily_input, instance_monthly_input, instance_yearly_input, all_prompts, instance_distances, instance_angularities)

            final_representation = final_representation.detach().cpu()
            instance_output = instance_output.detach().cpu()
            output = output.detach().cpu()

            output_loss = mae_error(output.reshape(-1,forecasting_window), instance_output.reshape(-1,forecasting_window))            

            actual_outputs.append(instance_output.reshape(1,-1,forecasting_window).cpu().detach().numpy())
            predicted_outputs.append(output.reshape(1,-1,forecasting_window).cpu().numpy())
            final_representations.append(final_representation.reshape(1,-1,8).cpu().numpy())
            
            print('Batch number :{} , loss : {}'.format(index+1, output_loss.item()))

    actual_outputs = np.array(actual_outputs).reshape(-1,len(all_indices),forecasting_window)
    predicted_outputs = np.array(predicted_outputs).reshape(-1,len(all_indices),forecasting_window)
    final_representations = np.array(final_representations).reshape(-1,len(all_indices),8)
    
    return (actual_outputs, predicted_outputs, final_representations)

def getInputsWithForecastedSWE(current_yr_index, dataset_input_index, loc_indices, historical_forecasted_swe, window = 1, step_size = 1):
    indices = sorted([dataset_input_index - (value * step_size) for value in range(0, window)])
    
    inputs = model_inputs[:,loc_indices,:][indices,:,:].reshape(window, len(loc_indices), -1)
    
    all_historical_available_swe =  torch.cat(historical_forecasted_swe,dim=0)

    all_historical_days_count = int(window * step_size)
    available_historical_days_count = all_historical_available_swe.shape[0]

    if available_historical_days_count <= all_historical_days_count:
        selected_swe_inputs = all_historical_available_swe[-available_historical_days_count:,:].reshape(available_historical_days_count, len(loc_indices), 1)
    else:
        selected_swe_inputs = all_historical_available_swe[-all_historical_days_count:,:].reshape(all_historical_days_count, len(loc_indices), 1)

    normalized_updated_swe_inputs = (selected_swe_inputs - min_swe)/(max_swe - min_swe)

    inputs[-len(selected_swe_inputs):, :, -1:] = normalized_updated_swe_inputs.to(dtype = torch.float32)

    return inputs

def getDailyPredictions(data_indices):
    actual_outputs = []
    collected_forecasted_outputs = []

    loss = nn.MSELoss()

    all_indices = sorted(list(train_loc_indices) + list(test_indices))

    for data_index in range(0,len(data_indices), forecasting_window):
        
        instance_daily_input  = getInputs(data_indices[data_index], all_indices, window = 1, step_size = 1)
        instance_monthly_input = getInputs(data_indices[data_index], all_indices, window = 120, step_size = 1)
        instance_yearly_input  = getInputs(data_indices[data_index], all_indices, window = 5, step_size = 365)

        instance_output = getOutputs(data_indices[data_index], all_indices, window = forecasting_window, step_size = forecasting_step_size)
        actual_outputs.append(instance_output.permute(1,0).reshape(forecasting_window, len(all_indices),1))

        instance_distances = torch.from_numpy(all_locations_distances[all_indices,:][:,all_indices]).to(device=device,dtype=torch.float32)
        instance_angularities = torch.from_numpy(all_locations_angularities[all_indices,:][:,all_indices]).to(device=device, dtype=torch.float32)

        instance_daily_input = torch.tensor(instance_daily_input).to(device=device,dtype=torch.float32)
        instance_monthly_input = torch.tensor(instance_monthly_input).to(device=device,dtype=torch.float32)
        instance_yearly_input = torch.tensor(instance_yearly_input).to(device=device,dtype=torch.float32)
                
        output, _ = swe_model(instance_daily_input, instance_monthly_input, instance_yearly_input, all_prompts, instance_distances, instance_angularities)
        output_loss = mae_error(output.reshape(-1,forecasting_window), instance_output.reshape(-1,forecasting_window))            

        output = output.detach().cpu().permute(0,2,1).reshape(forecasting_window,-1)
        collected_forecasted_outputs.append(output)

        print('Loss in day :{} , loss : {}'.format(data_index+1, output_loss.item()))

    all_predicted_outputs = torch.cat(collected_forecasted_outputs,dim=0).reshape(-1,len(all_indices),1).permute(1,0,2).detach().cpu().numpy()
    actual_daily_outputs  = torch.cat(actual_outputs, dim = 0).permute(1,0,2).detach().cpu().numpy() 

    return actual_daily_outputs, all_predicted_outputs

with open(output_2, 'w') as file1:
    with redirect_stdout(file1):
        print("Test Set 1")
        daily_window_actual_1, daily_window_predicted_1, test_1_final_representations  = TestModelSP(testing_yr_indices_1)
        daily_actual_1, daily_predicted_1 = getDailyPredictions(testing_yr_indices_1)
        print("================================================")

        print("Test Set 2")
        daily_window_actual_2, daily_window_predicted_2, test_2_final_representations = TestModelSP(testing_yr_indices_2)
        daily_actual_2, daily_predicted_2 = getDailyPredictions(testing_yr_indices_2)
        print("================================================")
        
        print("Test Set 3")
        daily_window_actual_3, daily_window_predicted_3, test_3_final_representations = TestModelSP(testing_yr_indices_3)
        daily_actual_3, daily_predicted_3 = getDailyPredictions(testing_yr_indices_3)
        print("================================================")
        
        print("Test Set 4")
        daily_window_actual_4, daily_window_predicted_4, test_4_final_representations = TestModelSP(testing_yr_indices_4)
        daily_actual_4, daily_predicted_4 = getDailyPredictions(testing_yr_indices_4)
        print("================================================")
        
        print("Test Set 5")
        daily_window_actual_5, daily_window_predicted_5, test_5_final_representations = TestModelSP(testing_yr_indices_5)
        daily_actual_5, daily_predicted_5 = getDailyPredictions(testing_yr_indices_5)
        print("================================================")

sys.stdout = original_stdout


joblib.dump(daily_window_predicted_1, '{}/results/predict/2014_daily_nva_window_predict.pkl'.format(baseUrl))
joblib.dump(daily_window_predicted_2, '{}/results/predict/2015_daily_nva_window_predict.pkl'.format(baseUrl))
joblib.dump(daily_window_predicted_3, '{}/results/predict/2016_daily_nva_window_predict.pkl'.format(baseUrl))
joblib.dump(daily_window_predicted_4, '{}/results/predict/2017_daily_nva_window_predict.pkl'.format(baseUrl))
joblib.dump(daily_window_predicted_5, '{}/results/predict/2018_daily_nva_window_predict.pkl'.format(baseUrl))

joblib.dump(daily_window_actual_1, '{}/results/actual/2014_daily_nva_window_actual.pkl'.format(baseUrl))
joblib.dump(daily_window_actual_2, '{}/results/actual/2015_daily_nva_window_actual.pkl'.format(baseUrl))
joblib.dump(daily_window_actual_3, '{}/results/actual/2016_daily_nva_window_actual.pkl'.format(baseUrl))
joblib.dump(daily_window_actual_4, '{}/results/actual/2017_daily_nva_window_actual.pkl'.format(baseUrl))
joblib.dump(daily_window_actual_5, '{}/results/actual/2018_daily_nva_window_actual.pkl'.format(baseUrl))

joblib.dump(daily_predicted_1, '{}/results/predict/2014_daily_nva_predict.pkl'.format(baseUrl))
joblib.dump(daily_predicted_2, '{}/results/predict/2015_daily_nva_predict.pkl'.format(baseUrl))
joblib.dump(daily_predicted_3, '{}/results/predict/2016_daily_nva_predict.pkl'.format(baseUrl))
joblib.dump(daily_predicted_4, '{}/results/predict/2017_daily_nva_predict.pkl'.format(baseUrl))
joblib.dump(daily_predicted_5, '{}/results/predict/2018_daily_nva_predict.pkl'.format(baseUrl))

joblib.dump(daily_actual_1, '{}/results/actual/2014_daily_nva_actual.pkl'.format(baseUrl))
joblib.dump(daily_actual_2, '{}/results/actual/2015_daily_nva_actual.pkl'.format(baseUrl))
joblib.dump(daily_actual_3, '{}/results/actual/2016_daily_nva_actual.pkl'.format(baseUrl))
joblib.dump(daily_actual_4, '{}/results/actual/2017_daily_nva_actual.pkl'.format(baseUrl))
joblib.dump(daily_actual_5, '{}/results/actual/2018_daily_nva_actual.pkl'.format(baseUrl))
