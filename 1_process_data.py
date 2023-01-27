"""
Script to process the transcript output from scrape.R
"""

import os
import pandas as pd
# internal imports
from params import vowels
from utils import tokenizer, extract_univocalic, load_training

###########################
# ---- (1) LOAD DATA ---- #

# Load the training data
lines = load_training()

# Convert to a dataframe
df = pd.DataFrame({'prompt':'', 'completion':lines})
# Save teh data
df.to_json(os.path.join('data','prompt_completion.jsonl'),orient='records',lines=True,force_ascii=False)


#############################################
# ---- (2) Create Univocalic dictonary ---- #

# (i) Extract all of the unique training data words
training_vocab = pd.Series(lines.str.lower().str.replace('[^a-z\\s]','',regex=True).str.split(' ').explode().unique())
training_vocab = training_vocab[training_vocab.str.len() > 0].to_list()

# (ii) Get all the words in the corpus that match the letters
tokens = pd.Series(list(tokenizer.vocab))
tokens = tokens.str.replace('Ġ','',regex=False)
tokens = tokens.str.lower()
tokens = tokens[~tokens.str.contains('[^a-z]')]
tokens = tokens.drop_duplicates().reset_index(drop=True)

# loop over each of the vowels
for vowel in vowels:
    # Extract from GPT tokens
    univocalic_tokens = extract_univocalic(vowel, tokens)
    # Extract from training data
    univocalic_training = extract_univocalic(vowel, training_vocab)
    univocalic_words = pd.Series(univocalic_tokens + univocalic_training).drop_duplicates()
    univocalic_words = univocalic_words.sort_values().reset_index(drop=True)
    univocalic_words.to_csv(os.path.join('data',f'univocalic_{vowel}.txt'),index=False)


print('~~~ End of 1_process_eunoia.py ~~~')