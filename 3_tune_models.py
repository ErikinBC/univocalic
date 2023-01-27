"""
Using the python API to prepare dataset and then tune models
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=4, help='Number of epochs to run')
parser.add_argument('--model', type=str, default='davinci', help='Which model should be used for fine-tuning? (default="davinci", options={ada, babbage, curie, davinci})')
args = parser.parse_args()
n_epochs = args.n_epochs
model = args.model

import os
import openai
import numpy as np
import pandas as pd
# Internal imports
from params import model_name, existing_model
from utils import set_openai_keys, wait_for_messages, get_embedding, get_finetuned_models

# Make sure keys are set
set_openai_keys()


###########################
# --- (1) UPLOAD DATA --- #

fname_data = 'training_data'
# Specify path to the jsonl file
path = os.path.join('data', f'{fname_data}.jsonl')
assert os.path.exists(path), f"File {path} does not exist"
uploaded_files = openai.File.list()['data']
# See if there is a filename "training_data"
uploaded_names = pd.Series([x['filename'] for x in uploaded_files])
has_data = uploaded_names.isin([fname_data]).any()
if not has_data:
    print('Uploading data...')
    data_upload = openai.File.create(file=open(path, "rb"), purpose='fine-tune', user_provided_filename=fname_data)
else:
    print('Data already uploaded')

# Get the id of the uploaded file
data_id = uploaded_files[uploaded_names.isin([fname_data]).idxmax()]['id']
print(f'Uploaded data has id: {data_id}')


#############################
# --- (2) EMBED DATASET --- #

# Set up save path
path_embeddings = os.path.join('data', f'{fname_data}_embeddings.npy')

if not os.path.exists(path_embeddings):
    print('Embedding data...')
    # Load the training data
    training_data = pd.read_json(path,orient='records',lines=True)
    lines = training_data['completion']
    n_lines = len(lines)
    # Embed the data
    embeddings = np.zeros((n_lines, 1536))
    for i, txt in enumerate(lines):
        print(f'Embedding line {i+1}/{n_lines}')
        txt = pd.Series(txt).str.replace('\n',' ',regex=True).str.strip()[0]
        embeddings[i,:] = get_embedding(txt)
    
    # Save the embeddings
    np.save(path_embeddings, embeddings)


#############################
# --- (3) TRAIN MODELS --- #

di_details = get_finetuned_models(model)
if model_name not in di_details:
    print(f'Training model {model_name}...')
    if existing_model is not None:
        if existing_model in di_details:
            print(f'Using existing model {existing_model} as starting point')
            model_id = di_details[existing_model]['fine_tuned_model']
        else:
            print(f'Existing model {existing_model} not found, starting from scratch with {model}')
            model_id = model
    finetune = openai.FineTune.create(training_file=data_id, model=model_id, n_epochs=n_epochs, suffix=model_name)
    # Retrieve the status
    openai_id = finetune['id']
    wait_for_messages(openai_id)
    input(f"Congratulations model {finetune['model']} has finished training, press Enter to continue...")
else:
    print('Model already trained')


print('~~~ End of 3_tune_models.py ~~~')