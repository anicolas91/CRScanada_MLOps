#!/bin/bash

# Define variables
CONDA_ENV_NAME="crs_env"

# Activate Conda environment
echo "Activating Conda environment: $CONDA_ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV_NAME

# Set environment variables
export AWS_ACCESS_KEY_ID=abc
export AWS_SECRET_ACCESS_KEY=xyz
export MLFLOW_S3_ENDPOINT_URL=http://localhost:4566 

# initialize docker containers for s3/db simulations
docker compose up db s3 -d --build

#sleep for a bit while docker thinks
sleep 10

# create an s3 bucket
aws --endpoint-url=http://localhost:4566 s3 mb s3://crs-data

echo "Setup complete."
# echo "To activate the Conda environment and set environment variables in your current session, run 'source setup_env.sh'."
