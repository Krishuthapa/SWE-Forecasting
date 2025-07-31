# SWE_Forecasting
ForeSWE: Forecasting Snow Water Equivalent with a Spatio-Temporal Attention-based Model

### Packages used.
Model implementation used Pytorch(v2.0.1) (LSTM and Attention models), GPyTorch (v1.12) (Gaussian process) packages. Data processing and visualization used multiple Python packages. We assume conda has been installed on the device running this code.

### Running Raw-GP model (after getting inside raw-gp folder)

#### For daily forecasting:
```
bash daily_gp_run.sh --base_url /path/to/SWE-Forecasting --env_name <your_choice_conda_env_name>
```

#### For weekly forecasting:
```
bash weekly_gp_run.sh --base_url /path/to/SWE-Forecasting --env_name <your_choice_conda_env_name>
```

### Running ForeSWE model (after getting inside raw-gp folder)

#### For training daily spatio-temporal attention model

```
bash att_daily_run.sh --base_url /path/to/SWE-Forecasting --env_name <your_choice_conda_att_env_name>
```

After training the spatio-temporal attention model, the representations from the model are further used to train the GP model in the output head and make predictions.

```
bash daily_foreswe_gp_run.sh --base_url /path/to/SWE-Forecasting --env_name <your_choice_conda_att_env_name>
```


#### For training weekly spatio-temporal attention model

```
bash att_weekly_run.sh --base_url /path/to/SWE-Forecasting --env_name <your_choice_conda_att_env_name>
```

After training the spatio-temporal attention model, the representations from the model are further used to train the GP model in the output head and make a prediction.

```
bash weekly_foreswe_gp_run.sh --base_url /path/to/SWE-Forecasting --env_name <your_choice_conda_att_env_name>
```
