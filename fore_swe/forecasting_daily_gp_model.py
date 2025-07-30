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
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy, MultitaskVariationalStrategy,LMCVariationalStrategy
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel, MultitaskKernel

from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import MultitaskGaussianLikelihood

torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    print("GPU is running.") 
    dev = "cuda:0" 
else: 
    print("CPU is running.")
    dev = "cpu" 

device = torch.device(dev)

seed_value = 50
torch.manual_seed(seed_value)

# --------------------------------------All Files Import-----------------------------------------
baseUrl = '</path/to/data_and_result_parent_dir>'

training_inputs = torch.load('{}/outputs/forecasting_modified_daily_att_inputs_for_gp_act.pt'.format(baseUrl))
training_outputs = torch.load('{}/outputs/forecasting_modified_daily_att_outputs_for_gp_act.pt'.format(baseUrl))

test_inputs_1 = torch.load('{}/outputs/daily_test1_inputs_for_gp_act.pt'.format(baseUrl))
test_inputs_2 = torch.load('{}/outputs/daily_test2_inputs_for_gp_act.pt'.format(baseUrl))
test_inputs_3 = torch.load('{}/outputs/daily_test3_inputs_for_gp_act.pt'.format(baseUrl))
test_inputs_4 = torch.load('{}/outputs/daily_test4_inputs_for_gp_act.pt'.format(baseUrl))
test_inputs_5 = torch.load('{}/outputs/daily_test5_inputs_for_gp_act.pt'.format(baseUrl))

test_outputs_1 = joblib.load('{}/results/actual/2014_daily_window_actual.pkl'.format(baseUrl))
test_outputs_2 = joblib.load('{}/results/actual/2015_daily_window_actual.pkl'.format(baseUrl))
test_outputs_3 = joblib.load('{}/results/actual/2016_daily_window_actual.pkl'.format(baseUrl))
test_outputs_4 = joblib.load('{}/results/actual/2017_daily_window_actual.pkl'.format(baseUrl))
test_outputs_5 = joblib.load('{}/results/actual/2018_daily_window_actual.pkl'.format(baseUrl))

training_inputs = torch.cat(training_inputs,dim = 0).reshape(-1,8).contiguous()
training_outputs = torch.cat(training_outputs, dim = 0).reshape(-1,10).contiguous()

train_min = training_inputs.min(dim=0, keepdim=True).values
train_mean = training_inputs.mean(dim=0, keepdim=True)
train_std = training_inputs.std(dim=0, keepdim=True)
train_max = training_inputs.max(dim=0, keepdim=True).values

training_inputs = training_inputs.numpy()
training_outputs = training_outputs.numpy()

print("Training input shape", training_inputs.shape)
print("Test shape", test_inputs_1.shape)

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
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([self.num_latents], ard_num_dims=8))+
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

all_inputs = torch.from_numpy(training_inputs).to(device=device, dtype= torch.float64)
all_outputs = torch.from_numpy(training_outputs).to(device=device, dtype= torch.float64)

train_dataset = TensorDataset(all_inputs, all_outputs)
train_loader = DataLoader(train_dataset, batch_size=5196, shuffle=True)

# initialize likelihood and model
model = MultitaskGPModel(num_tasks = 10, num_latents = 5, n_inducing_points = 7000, input_dim = 8)
likelihood = MultitaskGaussianLikelihood(num_tasks= 10)

model = model.to(device=device)
likelihood = likelihood.to(device=device)

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.12)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=all_inputs.size(0))

def train():
    model.train()
    for i in range(5):
        for batch_index, (x_batch, y_batch) in enumerate(train_loader):
            
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            print("Loss value is : {} in epoch:{} and batch:{}".format(loss.item(), i, batch_index))

train()

model.eval()
likelihood.eval()

def getTestsResults(test_loader):
    means = []
    low_values = []
    high_values = []

    mseLoss = nn.MSELoss()

    with torch.no_grad():
        for batch_index, (x_batch, y_batch) in enumerate(test_loader):
        
            preds = model(x_batch)
            loss = mseLoss(preds.mean, y_batch)

            lower, upper = preds.confidence_region()

            means.append(preds.mean.cpu().numpy())
            low_values.append(lower.cpu().numpy())
            high_values.append(upper.cpu().numpy())

    means = np.concatenate(means, axis = 0).reshape(-1,512,10)
    low_values = np.concatenate(low_values, axis = 0).reshape(-1,512,10)
    high_values = np.concatenate(high_values, axis = 0).reshape(-1,512,10)

    selected_means = []
    selected_low_values = []
    selected_high_values = []

    for index in range(0,len(means), 10):
        filtered_means = means[index,:,:].transpose(1,0).reshape(10, -1,1)
        filtered_low_values = low_values[index,:,:].transpose(1,0).reshape(10, -1,1)
        filtered_high_values = high_values[index,:,:].transpose(1,0).reshape(10, -1,1)

        selected_means.append(filtered_means)
        selected_low_values.append(filtered_low_values)
        selected_high_values.append(filtered_high_values)
    
    selected_means = np.concatenate(selected_means, axis = 0).transpose(1,0,2)
    selected_low_values = np.concatenate(selected_low_values, axis = 0).transpose(1,0,2)
    selected_high_values = np.concatenate(selected_high_values, axis = 0).transpose(1,0,2)

    return means, low_values, high_values, selected_means, selected_low_values, selected_high_values

test_inputs_1 = torch.tensor(test_inputs_1.reshape(-1,8)).contiguous()
test_inputs_2 = torch.tensor(test_inputs_2.reshape(-1,8)).contiguous()
test_inputs_3 = torch.tensor(test_inputs_3.reshape(-1,8)).contiguous()
test_inputs_4 = torch.tensor(test_inputs_4.reshape(-1,8)).contiguous()
test_inputs_5 = torch.tensor(test_inputs_5.reshape(-1,8)).contiguous()

test_inputs_1 = test_inputs_1.to(device=device)
test_inputs_2 = test_inputs_2.to(device=device)
test_inputs_3 = test_inputs_3.to(device=device)
test_inputs_4 = test_inputs_4.to(device=device)
test_inputs_5 = test_inputs_5.to(device=device)

test_outputs_1 = torch.tensor(test_outputs_1.reshape(-1,10)).contiguous().to(device=device)
test_outputs_2 = torch.tensor(test_outputs_2.reshape(-1,10)).contiguous().to(device=device)
test_outputs_3 = torch.tensor(test_outputs_3.reshape(-1,10)).contiguous().to(device=device)
test_outputs_4 = torch.tensor(test_outputs_4.reshape(-1,10)).contiguous().to(device=device)
test_outputs_5 = torch.tensor(test_outputs_5.reshape(-1,10)).contiguous().to(device=device)

test_dataset_1 = TensorDataset(test_inputs_1, test_outputs_1)
test_loader_1 = DataLoader(test_dataset_1, batch_size=8196, shuffle=False)

test_dataset_2 = TensorDataset(test_inputs_2, test_outputs_2)
test_loader_2 = DataLoader(test_dataset_2, batch_size=8196, shuffle=False)

test_dataset_3 = TensorDataset(test_inputs_3, test_outputs_3)
test_loader_3 = DataLoader(test_dataset_3, batch_size=8196, shuffle=False)

test_dataset_4 = TensorDataset(test_inputs_4, test_outputs_4)
test_loader_4 = DataLoader(test_dataset_4, batch_size=8196, shuffle=False)

test_dataset_5 = TensorDataset(test_inputs_5, test_outputs_5)
test_loader_5 = DataLoader(test_dataset_5, batch_size=8196, shuffle=False)

window_means_1, window_low_values_1, window_high_values_1, means_1, low_values_1, high_values_1 = getTestsResults(test_loader_1)
window_means_2, window_low_values_2, window_high_values_2, means_2, low_values_2, high_values_2 =getTestsResults(test_loader_2)
window_means_3, window_low_values_3, window_high_values_3, means_3, low_values_3, high_values_3 =getTestsResults(test_loader_3)
window_means_4, window_low_values_4, window_high_values_4, means_4, low_values_4, high_values_4 =getTestsResults(test_loader_4)
window_means_5, window_low_values_5, window_high_values_5, means_5, low_values_5, high_values_5 =getTestsResults(test_loader_5)

joblib.dump(window_means_1, '{}/results/predict/2014_daily_gp_window_predict.pkl'.format(baseUrl))
joblib.dump(window_means_2, '{}/results/predict/2015_daily_gp_window_predict.pkl'.format(baseUrl))
joblib.dump(window_means_3, '{}/results/predict/2016_daily_gp_window_predict.pkl'.format(baseUrl))
joblib.dump(window_means_4, '{}/results/predict/2017_daily_gp_window_predict.pkl'.format(baseUrl))
joblib.dump(window_means_5, '{}/results/predict/2018_daily_gp_window_predict.pkl'.format(baseUrl))

joblib.dump(window_low_values_1, '{}/results/predict/2014_daily_gp_low_window_predict.pkl'.format(baseUrl))
joblib.dump(window_low_values_2, '{}/results/predict/2015_daily_gp_low_window_predict.pkl'.format(baseUrl))
joblib.dump(window_low_values_3, '{}/results/predict/2016_daily_gp_low_window_predict.pkl'.format(baseUrl))
joblib.dump(window_low_values_4, '{}/results/predict/2017_daily_gp_low_window_predict.pkl'.format(baseUrl))
joblib.dump(window_low_values_5, '{}/results/predict/2018_daily_gp_low_window_predict.pkl'.format(baseUrl))

joblib.dump(window_high_values_1, '{}/results/predict/2014_daily_gp_high_window_predict.pkl'.format(baseUrl))
joblib.dump(window_high_values_2, '{}/results/predict/2015_daily_gp_high_window_predict.pkl'.format(baseUrl))
joblib.dump(window_high_values_3, '{}/results/predict/2016_daily_gp_high_window_predict.pkl'.format(baseUrl))
joblib.dump(window_high_values_4, '{}/results/predict/2017_daily_gp_high_window_predict.pkl'.format(baseUrl))
joblib.dump(window_high_values_5, '{}/results/predict/2018_daily_gp_high_window_predict.pkl'.format(baseUrl))

joblib.dump(means_1, '{}/results/predict/2014_daily_gp_predict.pkl'.format(baseUrl))
joblib.dump(means_2, '{}/results/predict/2015_daily_gp_predict.pkl'.format(baseUrl))
joblib.dump(means_3, '{}/results/predict/2016_daily_gp_predict.pkl'.format(baseUrl))
joblib.dump(means_4, '{}/results/predict/2017_daily_gp_predict.pkl'.format(baseUrl))
joblib.dump(means_5, '{}/results/predict/2018_daily_gp_predict.pkl'.format(baseUrl))

joblib.dump(low_values_1, '{}/results/predict/2014_daily_gp_lower_predict.pkl'.format(baseUrl))
joblib.dump(low_values_2, '{}/results/predict/2015_daily_gp_lower_predict.pkl'.format(baseUrl))
joblib.dump(low_values_3, '{}/results/predict/2016_daily_gp_lower_predict.pkl'.format(baseUrl))
joblib.dump(low_values_4, '{}/results/predict/2017_daily_gp_lower_predict.pkl'.format(baseUrl))
joblib.dump(low_values_5, '{}/results/predict/2018_daily_gp_lower_predict.pkl'.format(baseUrl))

joblib.dump(high_values_1, '{}/results/predict/2014_daily_gp_higher_predict.pkl'.format(baseUrl))
joblib.dump(high_values_2, '{}/results/predict/2015_daily_gp_higher_predict.pkl'.format(baseUrl))
joblib.dump(high_values_3, '{}/results/predict/2016_daily_gp_higher_predict.pkl'.format(baseUrl))
joblib.dump(high_values_4, '{}/results/predict/2017_daily_gp_higher_predict.pkl'.format(baseUrl))
joblib.dump(high_values_5, '{}/results/predict/2018_daily_gp_higher_predict.pkl'.format(baseUrl))
