"""
Look at statistics
"""

# Imports
import os
import numpy as np
import pandas as pd
import plotnine as pn
from mizani.formatters import percent_format
# Internal imports
from params import vowels, all_vowels, prompts, num_completions
from utils import tokenizer, load_training, extract_univocalic, get_embedding, cosine_similarity

###############################
# ---- (1) UNUSED TOKENS ---- #

# (i) Load training data
lines = load_training()
# Put to lower case and get all tokens
vocab = pd.Series(lines.str.lower().str.split().explode().unique())
vocab = pd.Series(vocab.str.replace('[^a-z]','', regex=True).unique())
vocab = vocab[vocab.str.len() > 0].reset_index(drop=True)
token_idx = tokenizer.encode(' '.join(vocab))
# Create a pd.Series of length n for a dtype=str
token_training = pd.Series([tokenizer.decode(idx) for idx in token_idx])

# (ii) Save to a file
for vowel in vowels:
    # Load univocalic words
    path = os.path.join('data',f'univocalic_{vowel}.txt')
    tokens_all = pd.read_csv(path).iloc[:,0].dropna()
    # Extract tokens from training that match letter
    tokens_training = pd.Series(extract_univocalic(vowel, vocab))
    tokens_training = tokens_training.sort_values().reset_index(drop=True)
    # Add on an "s", "ing", and remove plural
    tokens_training = pd.concat((tokens_training + 's', tokens_training + 'ing', tokens_training.str.replace('s$','', regex=True), tokens_training))
    # Find the unused tokens
    new_training = pd.Series(np.setdiff1d(tokens_all, tokens_training))
    # Save to file
    path = os.path.join('data',f'unused_{vowel}.txt')
    new_training.to_csv(path, index=False, header=False)


##################################
# ---- (2) COMPARING EPOCHS ---- #

# Merge the different epoch data together
path_epoch4 = os.path.join('output', 'prompt_results.csv')
path_epoch8 = os.path.join('output', 'prompt_results8.csv')
if os.path.exists(path_epoch4) and os.path.exists(path_epoch8):
    output_epoch4 = pd.read_csv(path_epoch4,header=None)
    output_epoch8 = pd.read_csv(path_epoch8,header=None)
    output_epoch = pd.concat([output_epoch4.assign(epoch=4), output_epoch8.assign(epoch=8)]).reset_index(drop=True).rename(columns={0:'txt'})
    del output_epoch4, output_epoch8
    # Add on the prompt
    output_epoch['prompt'] = np.tile(np.repeat(prompts, num_completions),2)
    # Get the vowel and the prompt
    output_epoch['vowel'] = output_epoch['txt'].str.split(' ', expand=True)[0].str.lower().str.replace('[^aeiou]','', regex=True).str[:1]
    # Add on the completion number
    output_epoch['num'] = output_epoch.groupby(['epoch','vowel','prompt']).cumcount() + 1
    output_epoch['txt'] = output_epoch['txt'].str.replace('\\n','',regex=True)

    # Split on sentences
    output_sentence = output_epoch.assign(txt=lambda x: x['txt'].str.split('\\.|\\?|\\!',regex=True)).explode('txt').reset_index(drop=True)
    output_sentence['txt'] = output_sentence['txt'].str.strip()
    output_sentence['sentence'] = output_sentence.groupby(['epoch','prompt','vowel','num']).cumcount()+1

    # Split on words
    output_words = output_sentence.assign(txt=lambda x: x['txt'].str.split('\\s+',regex=True)).explode('txt')
    output_words['txt'] = output_words['txt'].str.lower().str.replace('[^aeiou]','', regex=True).str[:1].str.strip()
    # Keep only vowels (remove consonant-only words)
    output_words = output_words[output_words['txt'].isin(all_vowels)].reset_index(drop=True)
    output_words = output_words.assign(violation=lambda x: x['txt'] == x['vowel'])
    output_words = output_words.groupby(['epoch','prompt','vowel','num','sentence'])['violation'].agg({'count','sum','mean'}).rename(columns={'count':'n_tot','sum':'n_valid','mean':'pct'})
    output_sentence = output_sentence.merge(output_words.reset_index(), 'inner')

    # Get the cumulative distribution for sentences with at least 10 words
    cn_gg = ['epoch','vowel']
    ecdf_sentence = output_sentence.query('n_tot >= 10').sort_values(cn_gg+['pct'])
    ecdf_sentence['ecdf'] = ecdf_sentence.groupby(cn_gg).cumcount()+1
    ecdf_sentence = ecdf_sentence.merge(ecdf_sentence.groupby(cn_gg)['ecdf'].max().reset_index().rename(columns={'ecdf':'n_emp'})).assign(ecdf=lambda x: x['ecdf']/x['n_emp'])
    ecdf_sentence['vowel'] = ecdf_sentence['vowel'].str.upper()
    # Print the summary statistics
    print('Percent of sentences with at least 10 words that failed:')
    print((100*(1-pd.concat([ecdf_sentence.assign(vowel='All'),ecdf_sentence]).groupby(cn_gg)['pct'].mean().round(2))).astype(int).astype(str)+'%')

    # Plot the ECDF using plotnine coloring by epoch
    gg = (pn.ggplot(ecdf_sentence, pn.aes(x='ecdf', y='pct', color='factor(epoch)')) + 
        pn.geom_line(size=0.5) + pn.theme_bw() +
        pn.scale_x_continuous(limits=(0,1),labels=percent_format()) + 
        pn.scale_y_continuous(limits=(0,1),labels=percent_format()) + 
        pn.facet_wrap('~vowel',labeller=pn.label_both) +
        pn.guides(color=pn.guide_legend(title='# of epochs')) + 
        pn.labs(y='% words match vowel restriction', x='Cumulative % of sentences'))
    gg.save(os.path.join('output','ecdf_epoch.png'), width=8, height=3)

    # Plot the ECDF but this time coloring by vowel
    gg = (pn.ggplot(ecdf_sentence, pn.aes(x='ecdf', y='pct', color='factor(vowel)')) + 
        pn.geom_line(size=0.5) + pn.theme_bw() +    
        pn.scale_x_continuous(limits=(0,1),labels=percent_format()) +
        pn.scale_y_continuous(limits=(0,1),labels=percent_format()) +
        pn.facet_wrap('~epoch',labeller=pn.label_both) +
        pn.guides(color=pn.guide_legend(title='Vowel')) +
        pn.labs(y='% words match vowel restriction', x='Cumulative % of sentences'))
    gg.save(os.path.join('output','ecdf_vowel.png'), width=8, height=3)


###################################
# ---- (3) SIMILAR SENTENCES ---- #

# Which sentences are most similar to training's works?
dat_training = pd.DataFrame({'txt':lines})
dat_training['vowel'] = dat_training['txt'].str.lower().str.split(' ',regex=False).str[0].str.replace('[^aeiou]','', regex=True).str[:1]
dat_training = dat_training[dat_training['vowel'].isin(vowels)]
# Split on sentences
dat_training = dat_training.assign(txt=lambda x: x['txt'].str.split('\\.|\\?|\\!',regex=True)).explode('txt').reset_index(drop=True)
dat_training['txt'] = dat_training['txt'].str.replace('\\(|\\)','',regex=True).str.strip()
dat_training['nwords'] = dat_training['txt'].str.split(' ',regex=False).str.len()
dat_training = dat_training.query('nwords >= 10').reset_index(drop=True)

# Loop throw the vowels and embed all sentences
for vowel in vowels:
    # Get the sentences
    sentences_training = dat_training.loc[dat_training['vowel']==vowel, 'txt'].to_list()
    sentences_gpt = output_sentence.loc[output_sentence['vowel']==vowel, 'txt'].to_list()
    # Embed the sentences
    training_embeddings = get_embedding(sentences_training)
    gpt_embeddings = get_embedding(sentences_gpt)
    # Compute the cosine similarity
    similiary = cosine_similarity(gpt_embeddings, training_embeddings.T)
    idx_max = np.argmax(similiary,1)
    res_vowel = pd.DataFrame({'sentence':sentences_gpt, 'closest':[sentences_training[i] for i in idx_max], 'cosine':similiary.max(1)}).sort_values('cosine',ascending=False)
    # Save the results
    res_vowel.to_csv(os.path.join('output','similar_sentences_'+vowel+'.csv'), index=False)

    





