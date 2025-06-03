#!/bin/bash
#SBATCH -J test   #jobname
#SBATCH --partition=gpu
#SBATCH --mem=6GB


cd $HOME/scratch/2025_STB/2025_cell_painting_sperm

conda activate python3.10


# install the aws to download

pip3 install awscli --upgrade --user

cd scratch

#  https://github.com/zaritskylab/AnomalyDetectionScreening
#  https://www.biorxiv.org/content/10.1101/2024.06.01.595856v1.full.pdf

aws s3 cp --no-sign-request --recursive s3://cellpainting-gallery/cpg0003-rosetta/broad/workspace/ ./



# 1. Clone and navigate
git clone https://github.com/zaritskylab/AnomalyDetectionScreening.git
cd AnomalyDetectionScreening

# 2. Create and activate environment
mamba create -n pytorch_anomaly python=3.10.9
conda activate pytorch_anomaly

# 3. Install dependencies
pip install -r requirements.txt

# 4. install the package in development mode
pip install -e .


# install pycyotminer from PyPI
pip install pycytominer


# How to use it .. 
# Train anomaly detection model
python main.py --flow train --exp_name <exp_name> --config configs/<config>.yaml

# Evaluate results (calculates replication %, MoA classification, SHAP explanations)
python main.py --exp_name <exp_name> --flow eval

