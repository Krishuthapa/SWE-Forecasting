import torch
import numpy as np
import pandas as pd

def getDevice():
    dev= 'cpu'

    if torch.cuda.is_available():
        print("GPU is running.") 
        dev = "cuda:0" 
    else: 
        print("CPU is running.")
        dev = "cpu" 

    return dev

def getInputs(model_inputs, index,loc_indices, window = 15, step_size = 1):
    indices = sorted([index - (value * step_size) for value in range(0, window)])

    inputs = model_inputs[:,loc_indices,:][indices,:,:].reshape(window, len(loc_indices), -1)

    return inputs.permute(1,0,2)

def getOutputs(model_outputs, index, loc_indices, window = 15, step_size = 1):
    indices = sorted([index + (value * step_size) for value in range(1, window+1)])

    outputs = model_outputs[:,loc_indices,:][indices,:,:].permute(1,0,2)

    return outputs.reshape(1,len(loc_indices), len(indices))

def getDayBreakdown(all_dates, index, loc_indices, past_window = 15, future_window = 0, step_size = 1):
    indices = sorted(value for value in range(index - (past_window * step_size), index + step_size * future_window, step_size))
    
    selected_dates = pd.to_datetime(all_dates[indices])

    days = selected_dates.day
    norm_day = (days - 1)/30

    weekdays = selected_dates.weekday
    norm_weekday = weekdays/6

    months = selected_dates.month
    norm_month = (months - 1)/11

    time_features = torch.from_numpy(np.stack([norm_day,norm_weekday,norm_month], axis = 1)).unsqueeze(1).repeat(1,len(loc_indices),1)

    return time_features.permute(1,0,2)

