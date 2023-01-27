"""
Compare the 'Russ-like' answers between models
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='davinci', help='Which model should be used for fine-tuning? (default="davinci", options={ada, babbage, curie, davinci})')
args = parser.parse_args()
model = args.model

import os
import openai
import numpy as np
import pandas as pd
# Cost data
from cost import custom
from utils import n_tokens, set_openai_keys, makeifnot, get_finetuned_models, get_embedding
from params import prompts, di_completion_params, prefix, model_name, num_completions, vowels, prefix

# Make the folder output if it doesn't exist
makeifnot('output')

# Set up keys
set_openai_keys()

# Set cost index to lower case
custom.index = custom.index.str.lower()


##########################################
# ---- (1) ESTIMATE INFERENCE COSTS ---- #

# Calculate the number of tokens in each prompt
n_tok_prompts = n_tokens(prefix) + np.array([n_tokens(p) for p in prompts])
# Calculate the total number of tokens including the response
n_tok_total = sum(n_tok_prompts + di_completion_params['max_tokens'])
total_cost_generation = custom.loc[model, 'Usage'] * n_tok_total * num_completions
print(f'Cost to generate all prompts: ${total_cost_generation:.2f}')


#############################
# ---- (2) RUN PROMPTS ---- #

# Get the name of the fine-tuned model
di_details = get_finetuned_models(model)
model_id = di_details[model_name]['fine_tuned_model']

# Create a temperature sequence to match number of completions
temperature_seq = np.linspace(0.8, 1, num_completions)

holder = []
for i, prompt in enumerate(prompts):
    print(f'Prompt {i+1} of {len(prompts)}')
    # Set up the completion parameters
    di_completion_params['model'] = model_id
    di_completion_params['prompt'] = prefix + ' ' + prompt
    for temperature in temperature_seq:
        di_completion_params['temperature'] = temperature
        # Run the query
        completion = openai.Completion.create(**di_completion_params)
        res = prompt + completion['choices'][0]['text']
        holder.append(res)
# Merge results into a dataframe
df = pd.DataFrame(holder)
path_prompt = os.path.join('output','prompt_results.csv')
df.to_csv(path_prompt,index=False, header=False, escapechar='\\')


##############################################
# ---- (3) GET EMBEDDINGS ON UNIVOCALIC ---- #

for vowel in vowels:
    print(f'Getting embeddings for vowel {vowel}')
    # (i) Load the univocalic words
    path_words = os.path.join('data',f'univocalic_{vowel}.txt')
    words = pd.read_csv(path_words)['0']
    words = words.dropna().reset_index(drop=True)
    n_words = len(words)
    # (ii) Get the embeddings (break into 1000 word chunks)
    chunk_size = 1000
    n_iter = int(np.ceil(n_words/chunk_size))
    embeddings = np.zeros((n_words, 1536))
    for j in range(n_iter):
        print(f'Chunk {j+1} of {n_iter}')
        istart, iend = j*chunk_size, (j+1)*chunk_size
        embeddings[istart:iend] = get_embedding(words[istart:iend].to_list())    
    # (iii) Save the embeddings
    df_embeddings = pd.DataFrame(embeddings.T,columns=words)
    df_embeddings.to_csv(os.path.join('output',f'embeddings_{vowel}.csv'),index=False)


print('~~~ END OF 4_prompt_baseline.py ~~~')