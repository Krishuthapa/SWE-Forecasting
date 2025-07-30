#!/bin/bash

export BASE_URL='</path/to/data_and_result_parent_dir>'
export FORECASTING_WINDOW=10  # 4 for weekly
export FORECASTING_STEP=1     # 7 for weekly
export INPUT_DIM=21

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gp_env

PYTHONPATH=<path_to_code_parent_dir>/raw_gp \
python <path_to_code_parent_dir>/raw_gp/gp_model_daily.py

conda deactivate

echo "Completed job on host $(hostname)"
