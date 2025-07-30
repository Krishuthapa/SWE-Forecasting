import torch
import torch.nn as nn

import numpy as np
import pandas as pd

import random

import os
import sys

import math

import time
import joblib

import gpytorch

from gpytorch.models import ApproximateGP
from torch.utils.data import TensorDataset, DataLoader

from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from helper_functions import getDevice, getOutputs
import gp_dataloader as DL

torch.set_default_dtype(torch.float64)

############################################# Some initial setups here ###################################

device = torch.device(getDevice())

BASE_URL = os.getenv("BASE_URL")
INPUT_DIM = int(os.getenv("INPUT_DIM"))

FORECASTING_WINDOW = int(os.getenv("FORECASTING_WINDOW"))
FORECASTING_STEP_SIZE = int(os.getenv("FORECASTING_STEP"))

test_indices = [43, 66, 338, 463, 115, 105, 83, 306, 481, 21, 185, 136, 110, 406, 408, 132, 
                120, 197, 7, 301, 375, 236, 269, 287, 49, 241, 212, 145, 264, 435, 73, 277, 364, 
                438, 372, 368, 243, 493, 497, 14, 32, 433, 313, 213, 42, 475, 160, 190, 281, 285, 
                316, 461, 154, 424, 310, 340, 179, 167, 193, 466, 56, 158, 365, 492, 432, 415, 68, 
                409, 76, 148, 267, 227, 91, 189, 472, 162, 147, 395, 113, 2, 271, 357, 347, 335, 507, 
                465, 57, 19, 442, 381, 398, 79, 337, 501, 41, 339, 331, 63, 359, 232, 323, 258]

test_indices = sorted(test_indices)

############################################## Setting up dataloader ####################################
dataloader = DL.DataLoader(test_indices = test_indices, temp_window = 7)

model_inputs, model_outputs, all_dates = dataloader.getAllInputAndTargets()

train_loc_indices = dataloader.getTrainingIndices()
training_data_indices = dataloader.getTrainingDataIndices()
training_data_indices = np.array(training_data_indices)[:-(FORECASTING_WINDOW * FORECASTING_STEP_SIZE)]

testing_yr_indices_1 , testing_yr_indices_2, testing_yr_indices_3, testing_yr_indices_4, testing_yr_indices_5  = dataloader.getTestingDataIndices()

# Model Inputs and Outputs.
model_inputs = torch.from_numpy(model_inputs)
model_outputs = torch.from_numpy(model_outputs)

max_swe,min_swe = dataloader.getMaxMinSWE()
#########################################################################################################

def get_inputs_and_outputs(all_inputs, all_outputs, data_indices, location_indices):
    accumulated_inputs = []
    accumulated_outputs = []

    for row_index in data_indices:
        row_inputs = all_inputs[row_index,location_indices,:].reshape(1,len(location_indices),-1)
        row_outputs = getOutputs(all_outputs,row_index,location_indices,FORECASTING_WINDOW,FORECASTING_STEP_SIZE)

        accumulated_inputs.append(row_inputs)
        accumulated_outputs.append(row_outputs)

    accumulated_inputs = torch.cat(accumulated_inputs,dim=0)
    accumulated_outputs = torch.cat(accumulated_outputs,dim=0)

    return accumulated_inputs, accumulated_outputs

training_inputs, training_outputs = get_inputs_and_outputs(model_inputs, model_outputs, training_data_indices, sorted(list(train_loc_indices) + list(test_indices)))

test_inputs_1, test_outputs_1 = get_inputs_and_outputs(model_inputs, model_outputs, testing_yr_indices_1, sorted(list(train_loc_indices) + list(test_indices)))
test_inputs_2, test_outputs_2 = get_inputs_and_outputs(model_inputs, model_outputs, testing_yr_indices_2, sorted(list(train_loc_indices) + list(test_indices)))
test_inputs_3, test_outputs_3 = get_inputs_and_outputs(model_inputs, model_outputs, testing_yr_indices_3, sorted(list(train_loc_indices) + list(test_indices)))
test_inputs_4, test_outputs_4 = get_inputs_and_outputs(model_inputs, model_outputs, testing_yr_indices_4, sorted(list(train_loc_indices) + list(test_indices)))
test_inputs_5, test_outputs_5 = get_inputs_and_outputs(model_inputs, model_outputs, testing_yr_indices_5, sorted(list(train_loc_indices) + list(test_indices)))

# --------------------------------------All Files Import-----------------------------------------
baseUrl = '</path/to/data_and_result_parent_dir>'

training_inputs = training_inputs.contiguous()
training_outputs = training_outputs.contiguous()

class MultitaskGPModel(ApproximateGP):
    def __init__(self, num_latents = 4, num_tasks = 10, n_inducing_points = 5000, input_dim = 8):

        self.num_latents = num_latents
        self.num_tasks = num_tasks
        self.n_inducing_points = n_inducing_points
        self.input_dim = input_dim

        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(self.num_latents,self.n_inducing_points, input_dim)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([self.num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=self.num_tasks,
            num_latents=self.num_latents,
            latent_dim=-1,
            # rank = 2
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([self.num_latents]))
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([self.num_latents], ard_num_dims=INPUT_DIM))+
            # gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=torch.Size([self.num_latents])) +
            gpytorch.kernels.LinearKernel(batch_shape=torch.Size([self.num_latents])),
            batch_shape=torch.Size([self.num_latents])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

train_dataset = TensorDataset(training_inputs.reshape(-1,INPUT_DIM), training_outputs.reshape(-1,FORECASTING_WINDOW))
train_loader = DataLoader(train_dataset, batch_size=5196, shuffle=True)

# initialize likelihood and model
model = MultitaskGPModel(num_tasks = FORECASTING_WINDOW, num_latents = 5, n_inducing_points = 7000, input_dim = INPUT_DIM)
likelihood = MultitaskGaussianLikelihood(num_tasks= FORECASTING_WINDOW)

model = model.to(device=device)
likelihood = likelihood.to(device=device)

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.12)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=training_inputs.reshape(-1,INPUT_DIM).size(0))

def train():
    model.train()
    for i in range(4):
        for batch_index, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device=device, dtype= torch.float64)
            y_batch = y_batch.to(device=device, dtype=torch.float64)
            
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            print("Loss value is : {} in epoch:{} and batch:{}".format(loss.item(), i, batch_index))
 
            del x_batch, y_batch, output
            torch.cuda.empty_cache()
           
train()

model.eval()
likelihood.eval()

def getTestsResults(test_loader):
    means = []
    low_values = []
    high_values = []

    mseLoss = nn.MSELoss()

    print("============================================================")

    with torch.no_grad():
        for batch_index, (x_batch, y_batch) in enumerate(test_loader):
        
            preds = model(x_batch)
            loss = mseLoss(preds.mean, y_batch)

            print(f"Loss of {batch_index} is {loss}")

            lower, upper = preds.confidence_region()

            means.append(preds.mean.cpu().numpy())
            low_values.append(lower.cpu().numpy())
            high_values.append(upper.cpu().numpy())

    means = np.concatenate(means, axis = 0).reshape(-1,512,FORECASTING_WINDOW)
    low_values = np.concatenate(low_values, axis = 0).reshape(-1,512,FORECASTING_WINDOW)
    high_values = np.concatenate(high_values, axis = 0).reshape(-1,512,FORECASTING_WINDOW)

    selected_means = []
    selected_low_values = []
    selected_high_values = []

    for index in range(0,len(means), FORECASTING_WINDOW * FORECASTING_STEP_SIZE):
        filtered_means = means[index,:,:].transpose(1,0).reshape(FORECASTING_WINDOW, -1,1)
        filtered_low_values = low_values[index,:,:].transpose(1,0).reshape(FORECASTING_WINDOW, -1,1)
        filtered_high_values = high_values[index,:,:].transpose(1,0).reshape(FORECASTING_WINDOW, -1,1)

        selected_means.append(filtered_means)
        selected_low_values.append(filtered_low_values)
        selected_high_values.append(filtered_high_values)
    
    selected_means = np.concatenate(selected_means, axis = 0).transpose(1,0,2)
    selected_low_values = np.concatenate(selected_low_values, axis = 0).transpose(1,0,2)
    selected_high_values = np.concatenate(selected_high_values, axis = 0).transpose(1,0,2)

    return means,low_values, high_values, selected_means, selected_low_values, selected_high_values

def run_and_save_result(inputs, outputs, year):
    inputs = inputs.reshape(-1,INPUT_DIM).contiguous().to(device=device)
    outputs = outputs.reshape(-1,FORECASTING_WINDOW).contiguous().to(device=device)

    dataset = TensorDataset(inputs,outputs)
    dataloader = DataLoader(dataset, batch_size=8196, shuffle=False)

    window_means, window_low, window_high, means, low_values, high_values = getTestsResults(dataloader)

    joblib.dump(window_means, '{}/results/predict/{}_weekly_raw_gp_w_predict.pkl'.format(baseUrl, year))
    joblib.dump(window_low, '{}/results/predict/{}_weekly_raw_gp_w_low_predict.pkl'.format(baseUrl, year))
    joblib.dump(window_high, '{}/results/predict/{}_weekly_raw_gp_w_high_predict.pkl'.format(baseUrl, year))
    joblib.dump(means, '{}/results/predict/{}_weekly_raw_gp_predict.pkl'.format(baseUrl,year))
    joblib.dump(low_values, '{}/results/predict/{}_weekly_raw_gp_lower_predict.pkl'.format(baseUrl, year))
    joblib.dump(high_values, '{}/results/predict/{}_weekly_raw_gp_higher_predict.pkl'.format(baseUrl,year))

    del inputs, outputs, dataset, dataloader
    del window_means, means, low_values, high_values
    torch.cuda.empty_cache()

run_and_save_result(test_inputs_1,test_outputs_1,2014)
run_and_save_result(test_inputs_2,test_outputs_2,2015)
run_and_save_result(test_inputs_3,test_outputs_3,2016)
run_and_save_result(test_inputs_4,test_outputs_4,2017)
run_and_save_result(test_inputs_5,test_outputs_5,2018)
