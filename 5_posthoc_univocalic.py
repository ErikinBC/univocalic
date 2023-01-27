"""
Force poems to be univocalic by replacing words with synonyms or words in a similar embedding space that are univocalic
"""

# Load modules
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
# Load utils
from utils import mask_univocalic, get_embedding, cosine_similarity

# Number of words to present
k = 4


###########################
# ---- (1) LOAD DATA ---- #

# (i) Load the prompt results
path_prompt = os.path.join('output','prompt_results.csv')
dat_prompts = pd.read_csv(path_prompt, header=None, escapechar='\\')
dat_prompts.rename(columns={0:'prompt'}, inplace=True)
#dat_prompts.rename_axis('idx', inplace=True)
# Get the vowel
dat_prompts['vowel'] = dat_prompts['prompt'].str.split('\\s',1,expand=True)[0].str.lower().str.replace('[^aeiou]','',regex=True).str[:1]

# (ii) For each vowel and prompt, load the associated embedding and identify words that fail the univocalic test
holder = []
for vowel, g in dat_prompts.groupby(['vowel']):
    print(f'Processing vowel: {vowel}')
    prompts = g['prompt'].copy().rename_axis('idx')
    # (i) Load the embeddings
    path_embed = os.path.join('output',f'embeddings_{vowel}.csv')
    dat_embed = pd.read_csv(path_embed)
    
    # (ii) Identify words that fail the univocalic test. Split on any reasonalbe separator
    all_prompts = prompts.str.split('([^a-zA-Z0-9])',regex=True).explode().reset_index()
    # Identify prompts that start with a capital letter
    all_prompts['is_cap'] = all_prompts['prompt'].str.contains('^[A-Z]')
    # Set prompts to lower case
    all_prompts['prompt'] = all_prompts['prompt'].str.lower()
    # Determine which words fail the univocalic test
    masked_prompts = mask_univocalic(vowel, all_prompts['prompt'])
    # check they are the same length
    assert len(all_prompts) == len(masked_prompts), f"Lengths do not match"
    # Add word order
    all_prompts['wordnum'] = all_prompts.groupby('idx').cumcount()
    all_prompts['fails'] = masked_prompts == 'X'
    # Get the preceesing and succeeding words (not punctuation)
    all_prompts['is_word'] = ~all_prompts['prompt'].str.contains('[^a-zA-Z]')
    tmp = all_prompts.query('is_word').assign(pre=lambda x: x.groupby('idx')['prompt'].shift(1), post=lambda x: x.groupby('idx')['prompt'].shift(-1))[['idx','wordnum','pre','post']]
    all_prompts = all_prompts.merge(tmp, 'left').fillna('')

    # (iii) Extract unique failure words
    fails = list(all_prompts.query('is_word & fails')['prompt'].unique())

    # (iv) Get the failure embeddings
    fails_embed = get_embedding(fails)
    
    # (v) Get the top k words for each
    angle = cosine_similarity(fails_embed, dat_embed)
    idx = rankdata(-angle, axis=1)
    top_replacements = pd.DataFrame(np.tile(dat_embed.columns, [fails_embed.shape[0],1])[np.where(idx <= k)].reshape(-1,k)).apply(list,1)
    di_replacements = dict(zip(fails, top_replacements))
    explode_prompts = all_prompts.copy()
    explode_prompts['options'] = explode_prompts['prompt'].map(di_replacements)
    # Remove options that much the pre of post
    explode_prompts = explode_prompts.explode('options').query('~((options == pre) | (options == post))').drop(columns=['pre','post'])
    explode_prompts['options'].fillna('', inplace=True)
    # Merge back
    explode_prompts = explode_prompts.groupby(['idx','wordnum'])['options'].apply(lambda x: f"[{'|'.join(x)}]").replace('[]','',regex=False).reset_index()
    explode_prompts = all_prompts.drop(columns=['pre','post']).merge(explode_prompts)
    # If is_cap==True, capitalize the first letter
    explode_prompts = explode_prompts.assign(prompt = lambda x: np.where(x['is_cap'], x['prompt'].str.capitalize(), x['prompt']))
    # If there is an option, replace the failing word
    explode_prompts = explode_prompts.assign(prompt = lambda x: np.where(x['options'] != '', '{'+x['prompt']+'}'+x['options'], x['prompt'])).drop(columns='options')
    explode_prompts = explode_prompts.groupby('idx').apply(lambda x: ''.join(x['prompt']))

    # (vi) Save for later
    assert(explode_prompts.index == g.index).all(), "Indices do not match"
    res = g[['vowel', 'prompt']].assign(options=explode_prompts)
    # Count the number of suggestions
    res['num_suggestions'] = res['options'].str.count('\{')
    res = res.sort_values('num_suggestions').drop(columns='num_suggestions')
    # Remove any line breaks
    res = res.apply(lambda x: x.str.replace('\n',' ',regex=False))
    path_out = os.path.join('output',f'prompts_{vowel}.csv')
    res.to_csv(path_out, index=False)
