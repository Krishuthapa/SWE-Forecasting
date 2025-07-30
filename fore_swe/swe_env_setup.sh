#!/bin/bash

module load anaconda3

conda remove -n swe_env --all -y || echo "No existing environment to remove"
conda clean --all -y

conda create -n swe_env python=3.8 -y

source activate swe_env

pip cache purge

pip install scikit-learn matplotlib pandas sentence-transformers
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

python -c "import torch; print(torch.cuda.is_available())"

conda deactivate
