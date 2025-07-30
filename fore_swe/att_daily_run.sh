#!/bin/bash

# Default values
base_url="/path/to/base"         
env_name="foreswe_att_env"
requirements_file="requirement_files/foreswe_requirement.txt"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --base_url)
            base_url="$2"
            shift 2
            ;;
        --env_name)
            env_name="$2"
            shift 2
            ;;
        *)
            echo "❌ Unknown parameter: $1"
            exit 1
            ;;
    esac
done

export BASE_URL="$base_url"

echo "Using base_url=$base_url"
echo "Using env_name=$env_name"

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    echo "Conda environment '$env_name' exists. Removing it to start fresh..."
    conda env remove -n "$env_name" -y
fi

echo "Creating conda environment '$env_name'..."
conda create -y -n "$env_name" python=3.8
conda activate "$env_name"

echo "Installing dependencies from $base_url/$requirements_file"
pip install -r "$base_url/$requirements_file"
conda deactivate


conda activate "$env_name"

python -c "import torch; print(torch.__version__)"

export PYTHONPATH="$base_url"
python "$base_url/fore_swe/forecasting_att_daily_model.py"

conda deactivate

echo "✅ Completed job on host $(hostname)"
