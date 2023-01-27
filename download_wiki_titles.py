"""
Download wikipedia titles and find any univocalics
"""

# Load modules
import os
import requests
import numpy as np
import pandas as pd
from time import sleep
# Internal imports
from params import vowels
from utils import makeifnot, extract_univocalic

makeifnot('wiki')

# Define the base URL for the Wikipedia API
base_url = "https://en.wikipedia.org/w/api.php"

aplimit = 500
params = {
    "action": "query",
    "format": "json",
    "list": "allpages",
    "aplimit": str(aplimit),
    "apcontinue": "",
    "apfilterredir": "nonredirects",
}
chunksize = 100000


#################################
# ---- (1) DOWNLOAD TITLES ---- #

# Initialize variables
continue_extracting = True
i, j, alphabetical_titles = 0, 0, []
n_total = 0
# Number before being saved
while continue_extracting:
    i += 1
    if i % 10 == 0:
        print(f"Chunk {i}")
    # Define the parameters for the API call
    params["apfrom"] = alphabetical_titles[-1] if alphabetical_titles else ""
    # Make the API call
    response = requests.get(base_url, params=params)
    data = response.json()
    # Extract the page titles from the API response
    titles = [page["title"] for page in data["query"]["allpages"]]
    # Append the titles to the alphabetical_titles list
    alphabetical_titles += titles
    # Check if there are more pages to be extracted
    if "continue" in data:
        params["apcontinue"] = data["continue"]["apcontinue"]
    else:
        continue_extracting = False
    # Cool down time
    cooldowntime = np.random.uniform(0.05, 0.2)
    sleep(cooldowntime)
    # Save the titles every k'th chunk
    n_titles = len(alphabetical_titles)
    if (n_titles >= chunksize) or (continue_extracting==False):
        j += 1
        n_total += n_titles
        print(f"Saving {n_titles} titles (out of {n_total}) from {i} iterations")
        pd.Series(alphabetical_titles).to_csv(f'wiki/titles_{j}.txt', index=False, header=False)
        i = 0
        alphabetical_titles = [alphabetical_titles[-1]]


##################################
# ---- (2) FIND UNIVOCALICS ---- #

# Get list of txt files in the wiki folder
files = pd.Series(os.listdir('wiki'))
files = files[files.str.contains('\\.txt$',regex=True)]

# Loop through each file and append it to the holder
holder = []
for file in files:
    path = os.path.join('wiki', file)
    holder.append(pd.read_csv(path,header=None)[0])
# Merge all files
titles = pd.concat(holder).sort_values().reset_index(drop=True)
# Remove missing values and strings of length 0/1
titles = titles[titles.str.len()>1].dropna()
# Keep only entrees with some letters
titles = titles[titles.str.contains('[a-zA-Z]',regex=True)]
# Remove parantheses and brackets
titles = titles.str.replace('[\\(\\)\\[\\]]','',regex=True)
# Remove quotes
titles = titles.str.replace('"','',regex=True)
# Remove exlamation marks
titles = titles.str.replace('\\!','',regex=True)
# Remove white space
titles = titles.str.strip()
titles

# Split on spaces to get all title words
twords = titles.str.split(' ',regex=False).explode()
# Keep only entrees with some letters
twords = twords[twords.str.contains('[a-zA-Z]',regex=True)]
# Keep only words of length > 1 and remove duplicates
twords = twords[twords.str.len() > 1].drop_duplicates().dropna()
twords.reset_index(drop=True, inplace=True)

# Loop over the vowels and get the univocalics
for vowel in vowels:
    # Load the existing corpus
    univocalics_tokens = pd.read_csv(os.path.join('data',f'univocalic_{vowel}.txt')).iloc[:,0]
    # Map the non-standard characters to the matching standard characters
    univocalics_wiki = pd.Series(extract_univocalic(vowel, twords, set2lower=True))

    # Find the new univocalics (do not base on case)
    new_univocalics = univocalics_wiki[~univocalics_wiki.str.lower().isin(univocalics_tokens)]

