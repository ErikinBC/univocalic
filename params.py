"""
Stores the openai parameters
"""

# Will give fine-tuned model a custom name
model_name = 'univocalic_bot'
# Will be added before every prompt to alert inference of fine-tuning
prefix = 'Univocalic:'
# Can specify whether an existing model should be used (default=None)
existing_model = None

# Specify which vowels to use (vowels)
vowels = ['a','e','i']
# Do not adjust all_vowels unless you know what you're doing
all_vowels = ['a','e','i','o','u']

# Number of times the prompt will be queried
num_completions = 4

# Base OpenAI models (do not adjust models unless you know what you're doing)
models = {'ada':'text-ada-001',
          'babbage':'text-babbage-001',
          'curie':'text-curie-001', 
          'davinci':'text-davinci-003'}


# Set up the baseline parameter dictionary (model and prompt will need to swapped in)
di_completion_params = {
  "max_tokens": 172,
  'presence_penalty': 0.25,
  'frequency_penalty': 0.25,
  'n':1,
  "echo": False,
  "stream": False,
}

# Prompt list to test: 
prompts = [
  # A-prompts
  'An aardvark',
  'Can Hasan',
  'Hasan asks',
  'Hasan balks',
  'Hasan basks',
  'Hasan blasts',
  'Hasan can',
  'Hasan drafts',
  'Hasan fans',
  'Hasan gags',
  'Hasan has',
  'Hasan raps',
  'Hasan talks',
  'Hasan wants',  
  # E-prompts
  'Helen enters',
  'Helen feels',
  'Helen remembers',
  'Helen reveres',
  'Helen sees',
  'Helen sketches',
  'Restless Helen',
  'The Greeks',
  'Whenever Helen',
  'Whenever she sees',
  'Whenever she feels',
  # I-prompts
  'Drilling rigs',
  'Flirting prigs',
  'Gin licking',
  'In fighting',
  'Inhibiting minds',
  'Insisting Siri',
  'Itching pigs',
  'Limp wigs',
  'Lightning chills',
  'Ripping strings',
  'Silknit wigs',
  'Smirking misfits',
  'Stirring birch',
  'Stifling inscriptions',
  'Twinkling lights',
  ]


