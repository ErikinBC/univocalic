"""
Utility scripts
"""

import os
import openai
import numpy as np
import pandas as pd
from typing import Callable
from time import sleep
from datetime import datetime
from nltk.corpus import wordnet
local_tzname = datetime.now().astimezone().tzname()
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
# Internal imports
from params import models, all_vowels

def try2series(x) -> pd.Series:
    """Try to convert an object to a pandas series"""
    if not isinstance(x, pd.Series):
        if isinstance(x, list) and len(x) == 0:
            x = pd.Series(x, dtype='object')
        x = pd.Series(x)
    return x


def extract_univocalic(vowel:str, words:pd.Series, drop_y:bool=True, set2lower:bool=False) -> list:
    """
    Returns any words that contain the vowel and only that vowel
    """
    assert isinstance(vowel, str), f"Vowel must be a string"
    words = try2series(words)
    if set2lower:
        words = words.str.lower()
    if len(words) == 0:
        return []
    assert vowel in all_vowels, f"Vowel must be one of {all_vowels}"
    exclude = f"[{''.join([v for v in all_vowels if v != vowel])}]"
    idx_exclude = words.str.contains(exclude, regex=True, na=False)
    # Remove words that contain other vowels
    words = words[~idx_exclude]
    # Remove words that only contain consonants
    words = words[words.str.contains(vowel, regex=False, na=False)]
    # Remove words with a y
    if drop_y:
        words = words[~words.str.contains('y', regex=False, na=False)]
    # Remove white space
    words = words.str.strip()
    # Convert to list and return
    words = words.drop_duplicates().to_list()
    return words

def mask_univocalic(vowel:str, words:pd.Series) -> pd.Series:
    """
    Replaces any words that have an offending vowel with W
    """
    assert isinstance(vowel, str), f"Vowel must be a string"
    words = try2series(words)
    assert vowel in all_vowels, f"Vowel must be one of {all_vowels}"
    offending_vowels = list(np.setdiff1d(all_vowels, vowel))+['y']
    offending_regex = f"[{''.join(offending_vowels)}]"
    idx_replace = words.str.lower().str.contains(offending_regex)
    val_replace = 'X'+words.loc[idx_replace].str.replace('[a-zA-Z]*','',regex=True)
    masked_words = pd.Series(np.where(idx_replace, 'X'+words.str.replace('[a-zA-Z]*','',regex=True), words))
    masked_words.index = words.index
    return masked_words


def synonym_wordnet(txt:str) -> list:
    """Extract synonyms from a word using nltk wordnet"""
    assert isinstance(txt, str), f"Input must be a string"
    synonyms = []
    for syn in wordnet.synsets(txt):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms


def find_univocalic_synonyms(txt:str, vowel:str, extractor:Callable, verbose:bool=True) -> list:
    """
    Searches for univocalic synonyms of a word using a given extractor method. If no synonyms are found, the method will search for synonyms of the synonyms, and so on.
    
    Parameters
    ----------
    txt : str
        The word to search for synonyms
    vowel : str
        The vowel to search for
    extractor : Callable
        A function that takes a word and returns a list of synonyms
    
    Returns
    -------
    list
        A list of synonyms
    """
    # Input checks
    assert isinstance(txt, str), f"Input must be a string"
    assert isinstance(vowel, str), f"Vowel must be a string"
    assert isinstance(extractor, Callable), f"Extractor must be a function"    
    # Extract synonyms
    ready = True
    txt_list = [txt]
    i = 0
    while ready:
        i += 1
        words = list(set(sum([extractor(t) for t in txt_list],[])))
        uwords = extract_univocalic(vowel, words)
        if len(uwords) > 0:
            ready = False
        else:
            txt_list = words
        if i > 5:
            if verbose:
                print('Could not find any synonyms after 5 iterations')
            break
    return uwords


def makeifnot(path:str) -> None:
    """Make a directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)


def force2array(x) -> np.ndarray:
    """Force an object to be a numpy array"""
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return x

def array2flat(x:np.ndarray):
    """Force an array to be a flat array"""
    x = force2array(x)
    if len(x.shape) > 1:
        x = x.flatten()
    return x

def array2col(x:np.ndarray):
    """Force an array to be a column vector (if it's not already a matrix)"""
    x = force2array(x)
    if len(x.shape) == 1:
        x = x.reshape(-1,1)
    return x


def cosine_similarity(vec:np.ndarray, matrix:np.ndarray) -> np.ndarray:
    """
    Compare the cosine similarity of a vector to a matrix
    """
    vec = np.atleast_2d(vec)
    matrix = array2col(matrix)
    similiary = np.dot(vec, matrix) / (np.linalg.norm(vec, axis=1).reshape(-1,+1) * np.linalg.norm(matrix, axis=0).reshape(+1,-1))
    return similiary


def get_finetuned_models(model:str) -> dict:
    """
    Return a list of models that match a given name (e.g. davinci)
    """
    assert model in models, f"Model must be one of {models}"
    # Get a list of models that have been trained
    finetuned_cloud = openai.FineTune.list()['data']
    # Only look at models that succeesed
    finetuned_cloud = [x for x in finetuned_cloud if x['status'] == 'succeeded']
    finetuned_model = pd.Series([x['model'] for x in finetuned_cloud])
    finetuned_model = finetuned_model.str.split(':',regex=False).str[0]
    finetuned_names = pd.Series([x['fine_tuned_model'] for x in finetuned_cloud])
    # Find models that match the type we are looking for
    idx_model = finetuned_model == model
    # Return as a dict
    details = [finetuned_cloud[i] for i in np.where(idx_model)[0]]
    suffixes = finetuned_names[idx_model].str.split(':',regex=False).str[2].str.split('-',regex=False).str[0]
    di_ret = dict(zip(suffixes, details))
    return di_ret


def set_openai_keys() -> None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = os.getenv("OPENAI_ORG_ID")


def n_tokens(x:str) -> int:
    """Count the number of tokens in a string"""
    assert isinstance(x, str), 'Input must be a string'
    return len(tokenizer.encode(x))

def get_embedding(text:str or list, model="text-embedding-ada-002") -> np.ndarray:
    """
    Convert a string to an embedding using the OpenAI API

    Parameters
    ----------
    text : str
        The text to be converted to an embedding
    model : str, optional
        The model to be used for the conversion, by default "text-embedding-ada-002"

    For more model options see: https://beta.openai.com/docs/guides/embeddings
    """
    assert isinstance(text, str) or isinstance(text, list), 'Input must be a string or a list of strings'
    if isinstance(text, str):
        text = [text]
    text = pd.Series(text).str.replace("\n", "", regex=True).to_list()
    res = openai.Embedding.create(input=text, model=model)
    mat = np.vstack([r['embedding'] for r in res['data']])
    return mat


def wait_for_messages(openai_id:str, second_pause:int=30, terminal_message:str='Fine-tune succeeded') -> None:
    """
    Queries the openai.FineTune.list_events function to see which messages have been sent, and waits until the terminal message is sent
    """
    print(f'--- Waiting for messages from OpenAI for {openai_id} ---')
    keep_waiting = True
    # Loop until the terminal message is sent
    while keep_waiting:
        # Get the current time
        timerightnow = datetime.now().astimezone()
        # Get the messages
        df_messages = process_openai_messages(openai_id)
        # Print messages that have arrived since last update
        idx_print = (timerightnow >= df_messages['time']) & (df_messages['time'] + pd.DateOffset(seconds=second_pause) >= timerightnow)
        if len(idx_print) > 0:
            print('\n'.join(df_messages.loc[idx_print, 'message']))
        print(f'Time right now: {timerightnow}')
        # Check to see if the terminal message is in the messages
        keep_waiting = terminal_message not in df_messages['message'].values
        # Wait for a bit
        sleep(second_pause)


def process_openai_messages(openai_id:str) -> pd.DataFrame:
    """
    Process the messages from OpenAI
    """
    data = openai.FineTune.list_events(openai_id)['data']
    messages = pd.Series([x['message'] for x in data])
    date = pd.Series([x['created_at'] for x in data])
    date = pd.to_datetime(0, utc=True) + pd.TimedeltaIndex(date, unit='s')
    # Convert datetime from UTC to EST
    date = pd.Series(date).dt.tz_convert(local_tzname)
    # Convert to dataframe
    res = pd.DataFrame({'time':date,'message':messages})
    return res


def load_training() -> pd.Series:
    # Load data/training.txt line by line
    with open(os.path.join('data','training.txt'), 'r') as f:
        lines = pd.Series(f.readlines())
    # Remove the line break at the end of each chapter
    lines = lines.str.replace('\\n','',regex=True)
    # Keep only lines with at least one character
    lines = lines[lines.str.len() > 0]
    return lines