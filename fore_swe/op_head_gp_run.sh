#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gp_env

PYTHONPATH=<path_to_code_parent_dir>/ \
python <path_to_code_parent_dir>/fore_swe/forecasting_daily_gp_model.py
# python <path_to_code_parent_dir>/fore_swe/forecasting_weekly_gp_model.py

conda deactivate
echo "Completed job on host $(hostname)"
