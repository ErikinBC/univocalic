"""
Take output from process_transcripts and select "high quality" training samples and estimate cost
"""

# Set up argparse to include calculates for n_epochs, and max_cost
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=4, help='Number of epochs to run')
parser.add_argument('--max_cost', type=float, default=10, help='Maximum cost per model ($USD)')
parser.add_argument('--model', type=str, default='davinci', help='Which model should be used for fine-tuning? (default="davinci", options={ada, babbage, curie, davinci})')
args = parser.parse_args()
n_epochs = args.n_epochs
max_cost = args.max_cost
model = args.model

import os
import pandas as pd
from scipy import stats
# Internal imports
from utils import n_tokens
from cost import calculate_epoch_cost
from params import models, prefix, all_vowels
assert model in models, f'"{model}" is not a valid model. Please choose from {models}'


#########################
# --- (1) LOAD DATA --- #

# Load the prompt_completion JSONL data
prompt_completion = pd.read_json(os.path.join('data','prompt_completion.jsonl'), lines=True)
    
# Convert to a Pandas DataFrame
df = pd.DataFrame(prompt_completion).rename_axis('id').reset_index()
df = df.melt('id', var_name='pair', value_name='txt')
df = df.sort_values(['id','pair'],ascending=[True,False]).reset_index(drop=True)


##############################
# --- (2) SAVE FULL DATA --- #

training_data = df.copy()
idx_prompt = training_data['pair']=='prompt'
idx_completion = training_data['pair']=='completion'
# Add " Russ Roberts responds:" to the end of the prompt rows
training_data.loc[idx_prompt ,'txt'] = training_data.loc[idx_prompt ,'txt'] + ' ' + prefix
# Add a space to the beginning of the completion rows
training_data.loc[idx_completion ,'txt'] = ' ' + training_data.loc[idx_completion ,'txt']
# Add a suffix ending `\n` to all completions
training_data.loc[idx_completion ,'txt'] = training_data.loc[idx_completion ,'txt'] + '\n'


# Calculate the number of tokens found in the completion column using GPT2TokenizerFast
training_data['ntokens'] = training_data['txt'].apply(n_tokens)
# Calculate the number of words in the txt column
training_data['nwords'] = training_data['txt'].apply(lambda x: len(x.split()))
# Calculate the number of characters in the txt column
training_data['nchars'] = training_data['txt'].apply(lambda x: len(x))
# Print the average and total number of tokens, words, and characters
print(training_data.agg({'ntokens':['mean','sum'],'nwords':['mean','sum'],'nchars':['mean','sum']}).astype(int))

# Run a linear regression between number of words/characters and number of tokens
# Use scipy to do this
slope_tokens = stats.linregress(x=training_data['nwords'], y=training_data['ntokens'])[0]
print(f'For every one word, there is an average of {slope_tokens:.2f} tokens')
inv_slope_nchars = stats.linregress(x=training_data['ntokens'], y=training_data['nchars'])[0]
print(f'For every one token, there is an average of {inv_slope_nchars:.2f} characters')


##########################
# --- (3) CHECK COST --- #

# Calculate the (approximate) model specific cost for all epochs
n_tokens_training = training_data['ntokens'].sum()
dat_cost = calculate_epoch_cost(None, n_tokens_training)
dat_cost.index = dat_cost.index.str.lower()
cost_per_epoch = dat_cost.loc[model]
total_cost = cost_per_epoch * n_epochs
print(f'Cost per epoch: ${cost_per_epoch:.2f}, total cost: ${total_cost:.2f}')
total_cost_per_token = total_cost / n_tokens_training

if total_cost > max_cost:
    print(f'WARNING: The total cost of training (${total_cost:.2f}) is greater than the maximum cost allowed (${max_cost:.2f})\nWill apply a data reduction to reduce cost')
    # Determine the vowel used in each id
    vowel = training_data.loc[idx_completion, 'txt'].str.split(' ').str[1].str.lower().str.replace('[^aeiou]','',regex=True).str[0]
    assert vowel.isin(all_vowels).all(), 'Vowels are not all '
    # Add the vowels back on as a column in the training_data DataFrame
    training_data['vowel'] = vowel
    training_data['vowel'] = training_data['vowel'].fillna(method='bfill')
    assert (training_data.groupby('id')['vowel'].nunique() == 1).all(), 'Not all vowels are the same for each id'
    # Calculate the number of tokens for each id/vowel
    n_tokens_per_id = training_data.groupby(['id','vowel'])['ntokens'].sum().reset_index().sort_values(['vowel','ntokens']).reset_index(drop=True)
    # Add on the total cost for each id/vowel
    n_tokens_per_id['cost'] = n_tokens_per_id['ntokens'] * total_cost_per_token
    assert n_tokens_per_id['cost'].sum() == total_cost, 'Total cost does not match'
    # Find which index exceeds the maximum cost
    n_tokens_per_id['idx'] = n_tokens_per_id.groupby('vowel').cumcount()
    cumulative_cost = n_tokens_per_id.groupby('idx')['cost'].sum().cumsum().reset_index()
    cumulative_cost = cumulative_cost[cumulative_cost['cost'] < max_cost]
    assert cumulative_cost.shape[0] > 0, 'No ids are below the maximum cost'
    # Find the associated ids
    n_tokens_per_id = n_tokens_per_id[n_tokens_per_id['idx'] <= cumulative_cost['idx'].max()]
    # Keep only that training data
    training_data = training_data[training_data['id'].isin(n_tokens_per_id['id'])]
    training_data.reset_index(drop=True,inplace=True)

# Save to a JSONL file by pivoting on the id, pair, and completion columns
training_data_wide = training_data.pivot(index='id', columns='pair', values='txt')[['prompt','completion']]
training_data_wide.to_json(os.path.join('data','training_data.jsonl'),orient='records',lines=True, force_ascii=False)



print('~~~ End of 2_prepare_training.py ~~~')