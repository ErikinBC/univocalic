# Univocalic GPT_3

This purpose of this repo is to create a fine-tuned versions of the `davinci` [GPT-3 model](https://beta.openai.com/docs/models/gpt-3) to create [univocalic](https://en.wikipedia.org/wiki/Univocalic) sounding poetry creator (aka an antilipogrammatic constrained writer) in the style of Christian Bök (see [Eunoia](https://en.wikipedia.org/wiki/Eunoia_(book))). For a deeper dive in the technical results see [this post](https://www.erikdrysdale.com/univocalic), and to view the created poems [go here](https://bioeconometrician.github.io/univocalic).

## (1) Parameter and data configuration

### Parameters

The parameters of this pipeline are found in two places. In the [pipeline.sh](pipeline.sh) script, users can specify the number of epochs (`n_epochs`, default=4) and the maximum cost in USD (`max_cost`, default=5). If the amount of training tokens times the number of epochs exceeds the max costs, a fraction of the data will be used to ensure the training cost is bounded. Note that the [cost estimates](cost.py) are based on the OpenAI [pricing](https://openai.com/api/pricing/) as of Jan 27th, 2023.

The second set of parameters 

1. `model_name`: The name to assign to the model (default="univocalic_bot").
2. `prefix`: What prefix to add to all prompts (default="Univocalic:").
3. `existing_model`: Can be used to further train an existing model (default=None).
4. `vowels`: Which vowels to use for generation (currently set to only A, E, and I).
5. `num_completions`: Number of completions to retrieve for a single prompt (default=4).
6. `di_completion_params`: A dictionary of optional arguments that will be paseed into the `openai.Completion.create` function in the [prompt](4_prompt_baseline.py) script (`max_tokens`, `presence_penalty`, `frequency_penalty`, and `prompts`). See the API [documentation](https://beta.openai.com/docs/api-reference/completions/create) for more information.

### Data

Users should put any text excerpts they wish to use as training examples in the `data/training.txt` file. Excerpts of Eunoia can be found [here](http://poemsandpoetics.blogspot.com/2009/07/christian-bok-excerpts-from-eunoia.html).

### OpenAI account

To carry out model training and inference, users will need to have an account with OpenAI and several dollars of credit . See OpenAI's [documentation](https://beta.openai.com/docs/guides/fine-tuning) for more information about this. 

Users will need the following variables to the environment (I put it in their `.bash_profile` or `.bashrc`).

`export OPENAI_API_KEY="sk-...."`

`export OPENAI_ORG_ID="org-..."`

<br>

## (2) Running the pipeline

Before running the pipeline, ensure that you have a working version of Anaconda, and then set up the `univocalic` conda environment: `conda env create -f env.yml`. To run all the scripts call `pipeline.sh`. A description of each is listed below. 

* [1_process_data.py](1_process_data.py): i) Loads in the `data/training.txt` to create the JSONL-formatted data: `data/prompt_completion.jsonl` which will be used later, and ii) generates the `data/univocalic_{vowel}.txt` files which will form the foundation of the univocalic queries based on embeddings. 
* [2_prepare_training.py](2_prepare_training.py): Generates the training data (`data/training_data.jsonl`) that will cost at most `max_cost` USD to fine-tune.
* [3_tune_models.py](3_tune_models.py): i) Upload the training data to OpenAI, ii) calculate the emeddings for the training data (`data/training_data_embeddings.npy`), and iii) submit the fine-tuning order to OpenAI. Note that the script will not terminate until there is confirmation that the fine-tuning is complete.
* [4_prompt_baseline.py](4_prompt_baseline.py): i) Will run the `prompts` specified in the [params](params.py) file and will save the results to `output/prompt_results.csv`, and ii) will calculate embeddings for all words found in the `data/univocalic_{vowel}.txt` files and save them to `output/embeddings_{vowel}.csv`.
* [5_posthoc_univocalic](5_posthoc_univocalic): Will load the results from prompt_results.csv and provide word suggestions based on the embeddings from the previous step and store the results to `output/prompts_{vowel}.csv`.
* [6_statistics](6_statistics): i) Will see how many tokens were unused in your training data compared to the tokens and will store results to `data/unused_{vowel}.txt`, ii) Will compare whether the distribution of univocalic consistency differs between a different number of epochs and save figures to `output/ecdf_{epoch,vowel}.png` (however, users will need to manually specify `path_epoch{4,8}` in the script if the file names differ), and iii) Will match the sentences from the GPT-3 output to the training data using the Cosine similarity of the embeddings and save the results to `output/similar_sentences_{vowel}.csv`.

