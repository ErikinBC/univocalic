"""
Give word(s) and find top-k similar univocalics
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--vowel', type=str, help='Pick a vowel')
parser.add_argument('--words', nargs='+', help='List of word(s) to find similar univocalics for')
parser.add_argument('--k', type=int, default=4, help='Number of similar words to return (default: 4)')
# Add an optional flag to load the unused univocalic words
parser.add_argument('--unused', action='store_true', help='Load the unused univocalic words')
args = parser.parse_args()

# Check inputs
from utils import all_vowels
assert args.vowel in all_vowels, f"Vowel must be one of {all_vowels}"
assert isinstance(args.words, list), f"Words must be a list"
vowel = args.vowel
words = args.words
k = args.k
unused = args.unused

# Load external modules
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
# Load utils
from utils import cosine_similarity, get_embedding


# (i) Load the embeddings
file = 'embeddings'
if unused:
    file = 'unused'
path_embed = os.path.join('output',f'{file}_{vowel}.csv')
vowel_embeddings = pd.read_csv(path_embed)

# (ii) Generate embeddings for the words
word_embeddings = get_embedding(words)
n_words = len(words)

# (iii) Get the top k words for each
angle = cosine_similarity(word_embeddings, vowel_embeddings)
idx = rankdata(-angle, axis=1)
broadcast_columns = np.tile(vowel_embeddings.columns, [n_words,1])
broadcast_columns = broadcast_columns[np.where(idx <= k)].reshape(-1,k)
top_replacements = pd.DataFrame(broadcast_columns, index=words).apply(list,1)
print(top_replacements.to_dict())
