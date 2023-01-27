#!/bin/bash

# --- Pipeline to create EconChatR --- #

# Hard-coded parameters for model training
n_epochs=4  # Number of epochs to train the model
max_cost=5  # This is in USD
model="davinci"  # Model to use for fine-tuning

# (0) Set up conda environment if it doesn't already exist
# To build from scratch run: conda env create -f env.yml first
conda activate univocalic

# prepare data for cleaning
echo "--- (1) Running process_data.py --- "
python3 1_process_data.py
#   output: data/prompt_completion.jsonl

# Prepare the data for training
echo "--- (2) Running prepare_training.py --- "
python3 2_prepare_training.py --n_epochs $n_epochs --max_cost $max_cost --model $model
#   output: data/training_data.jsonl

# Use the OpenAI tools to train the model
echo "--- (3) Create fine-tuned model  --- "
python3 3_tune_models.py --n_epochs $n_epochs  --model $model
#   output: FINE-TUNES on the beta.openai.com website

# Comparing trained model results to existing
echo "--- (4) Run prompt_baseline  --- "
python3 4_prompt_baseline.py

# Get recommended univocalic words for the output
echo "--- (5) Run posthoc_univocalic.py  --- "
python3 5_posthoc_univocalic.py

# Calculate statistics
echo "--- (6) Run statistics.py  --- "
python3 6_statistics.py


echo "~~~ End of pipeline.sh ~~~"