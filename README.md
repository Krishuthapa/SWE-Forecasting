# Attention-based Models for Snow-Water Equivalent Prediction
## Krishu K Thapa, Supriya Savalkar, Bhupinderjeet Singh, Trong Nghia Hoang, Kirti Rajagopalan, Ananth Kalyanaraman
### Washington State University, Pullman, WA


#### Training Water Years: (1991 - 2014) and Testing Water Years: (2015-2019)

#### Packages used.
Model implementation used Pytorch(v2.0.1) (LSTM and Attention models), GPyTorch (v1.12) (Gaussian process) packages. Data processing and visualization used multiple Python packages. We assume conda has been installed on the device running this code.

#### Running Raw-GP model (after getting inside raw-gp folder)

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


### Output folder for predicted files (/outputs/predict/)
All the prediced SWE values are stored as .pkl files and can be found here corresponding to a model.

### Output folder for actual files (/outputs/actual/)
All the actual SWE values are stored as .pkl files and can be found here corresponding to a model.

## Citation

If you use our idea in your research, please cite:

Thapa, Krishu & Savalkar, Supriya & Singh, Bhupinderjeet & Trong Nghia Hoang & Rajagopalan, Kirti & Kalyanaraman, Ananth. (2026). ForeSWE: Forecasting Snow-Water Equivalent with an Uncertainty-Aware
Attention Model. Proceedings of the 40th AAAI Conference on Artificial Intelligence. arXiV: https://arxiv.org/pdf/2511.08856v1


