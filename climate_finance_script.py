# %%
import pandas as pd
import re

# %%
df=pd.read_csv('bln_reuters_climate_finance_2015to2020.csv')

# %%
italia = df[df['reporting_party']=='italy']

# %%
italia['assigned_geography'].unique()

# %%
info_ita = italia[italia['additional_information'].notna()].groupby("additional_information").first()

# %% [markdown]
# ## Bert

# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

descriptions = pd.Series(info_ita.index)

tokenizer_name = "ESGBERT/EnvironmentalBERT-action"
model_name = "ESGBERT/EnvironmentalBERT-action"
 
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, max_len=1024, padding=True, truncation=True)


pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0) 



# %%
# get token counts

tk_counts = []
for description in descriptions:
    tk_counts.append(len(tokenizer.tokenize(description)))
    

# %%
# plot token counts
import matplotlib.pyplot as plt
import seaborn as sns


sns.histplot(tk_counts, bins=20)

# %%
# n of tokencounts over 512
len([i for i in tk_counts if i > 512])

# %% [markdown]
# Good, very few texts are too long for the model

# %%
# empty cuda cache
import torch
torch.cuda.empty_cache()

# 

tokenizer_name = "ESGBERT/EnvironmentalBERT-environmental"
model_name = "ESGBERT/EnvironmentalBERT-environmental"
 
model_env = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer_env = AutoTokenizer.from_pretrained(tokenizer_name, max_len=512)
 
pipe_env = pipeline("text-classification", model=model_env, tokenizer=tokenizer_env, device=0) 

# %%

tests=descriptions.apply(lambda x: pipe(x, truncation=True))

# %%
tests_env=descriptions.apply(lambda x: pipe_env(x, truncation=True))

# %%
tests_env[0]

# %%
# unpack the results
results = [test[0] for test in tests]
# turn into a df with action and none columns, the value is the probability and the other is 1-probability
# get a list of the action scores
results_lists= list()


results_env= [test_env[0] for test_env in tests_env]

results_lists_env= list()

for i in range(len(results)):
    if results[i]['label']=='action' and results_env[i]['label']=='environmental':
        results_lists.append((1,1))
    elif results[i]['label']=='action' and results_env[i]['label']=='none':
        results_lists.append((0,1))
    elif results[i]['label']=='none' and results_env[i]['label']=='environmental':
        results_lists.append((1,0))
    else:
        results_lists.append((0,0))
        


# %%
results_lists = pd.DataFrame(results_lists, columns=['environmental', 'action'])

# %%
# join results_lists with descriptions
results_lists = pd.concat([descriptions,results_lists],axis=1)

# %%
# join results_lists with info_ita
info_ita_classified=info_ita.merge(results_lists, left_on='additional_information', right_on='additional_information')

# %%
# make environmental column the second column
cols = info_ita_classified.columns.tolist()
cols = cols[:1] + cols[-1:] + cols[1:-1]
info_ita_classified = info_ita_classified[cols]

# %%
# make action column the third column
cols = info_ita_classified.columns.tolist()
cols=cols[:2] + cols[-1:] + cols[2:-1]
info_ita_classified = info_ita_classified[cols]

# %%
info_ita_classified

# %%
# plot the results
import matplotlib.pyplot as plt

# bar plot of env 

fig, ax = plt.subplots(figsize=(10, 10))
# one bar for env 1 and one for env 0
ax.bar([0,1],info_ita_classified['environmental'].value_counts(), color=['green','grey'])
ax.set_xticks([0,1])
ax.set_xticklabels(['Environmental','Not Environmental'])
ax.set_ylabel('Number of projects')
ax.set_title('Environmental vs Not Environmental')
plt.show()

# %%
# bar plot of action    
fig, ax = plt.subplots(figsize=(10, 10))
# one bar for env 1 and one for env 0
ax.bar([0,1],info_ita_classified['action'].value_counts(), color=['blue','grey'])
ax.set_xticks([0,1])

# title and the rest
ax.set_xticklabels(['Action','No Action'])
ax.set_ylabel('Number of projects')
ax.set_title('Action vs No Action')
plt.show()


# %%
import numpy as np
# combined plot of action and env
fig, ax = plt.subplots(figsize=(10, 10))
# make a stacked bar plot, one bar for env 1 and one for env 0, with action and no action stacked on top
# df with intersection of action and env
df = info_ita_classified.groupby(['action','environmental']).size().reset_index().pivot(columns='action', index='environmental', values=0)
# stacked bar plot
df.plot(kind='bar', stacked=True, ax=ax, color=['grey','green'])

# title and the rest
ax.set_xticklabels(['Not Environmental','Environmental'])
# make labels horizontal
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_ylabel('Number of projects')
ax.set_title('Environmental and Action')
ax.legend(['No Action','Action'],loc='upper right')
plt.show()

# %%
# save dataset
info_ita_classified.to_csv('info_ita_classified.csv', index=False)

# %%
# save only the no env projects
info_ita_classified[info_ita_classified['environmental']==0].to_csv('info_ita_classified_none.csv', index=False)

# %% [markdown]
# ## Langchain script

# %%
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import torch

from transformers import StoppingCriteria, StoppingCriteriaList
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = []):
      StoppingCriteria.__init__(self), 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []):
      self.stops = stops
      for i in range(len(stops)):
        self.stops = self.stops[i]


# clear gpu memory
torch.cuda.empty_cache()
import gc
gc.collect()
from transformers import BitsAndBytesConfig


tokenizer = AutoTokenizer.from_pretrained("Upstage/SOLAR-10.7B-Instruct-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "Upstage/SOLAR-10.7B-Instruct-v1.0",
    torch_dtype=torch.float16,
    device_map="auto"
    )

stop_words_ids = [tokenizer.encode(stop_word) for stop_word in ["\n\n", "Human","\n\n\nHuman:", "uu","}","\n}\n\n", "\n}\n\n---\n\nNext project"]]
# unpack the list of lists
stop_words_ids = [item for sublist in stop_words_ids for item in sublist]

stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops = stop_words_ids)])     
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, do_sample=True, top_p=0.95, temperature=0.01, use_cache=False, stopping_criteria=stopping_criteria)
hf = HuggingFacePipeline(pipeline=pipe)

# %%


# load hf langchain interface


system_string = """You're working at the UN. You need to classify projects given to you in the following categories:

- an environmental project or green investment project (relevant_for_the_environment = yes)
- a project not relevant for the environment (relevant_for_the_environment = no)
- a project whose description is not clear enough to classify it (relevant_for_the_environment = unclear)
You give answers only in a Json format with three keys: reason (maximum 100 characters), relevant_for_the_environment. After the end of the first json, print uu.
"""

human_string = """Classify the following project:
{project_description}

Your answer in json:"""

system_prompt = SystemMessagePromptTemplate.from_template(system_string)
human_prompt = HumanMessagePromptTemplate.from_template(human_string)

chatpromt=ChatPromptTemplate.from_messages([system_prompt, human_prompt])

# Create a new chain
chain = LLMChain(llm=hf, prompt=chatpromt, llm_kwargs={"stop_sequences":["Human"]})

# %%
chain("Ciao")

# %%
# let's try
import pandas as pd
import os
descriptions = pd.Series(info_ita.index)

# use tqdm to track progress
from tqdm import tqdm

# iterate over descriptions, generate a response and save it in a list, save the list every 100 iterations
# check if resposnes.csv exists, if it does load it, if it doesn't create an empty list

if os.path.isfile('responses.jsonl'):
    responses = pd.read_json('responses.jsonl', lines=True).values.tolist()
else:
    responses = []
    
    
# suppress warnings
import warnings
warnings.filterwarnings('ignore')
    
for i in tqdm(range(len(descriptions)), desc='Generating responses', total=len(descriptions), unit='descriptions', leave=False):
    # pick up from where you left off
    if len(responses) > i:
        continue
    # generate a response checking if it is longer than 20 characters, also check if the response ends with "uu", check that response is a dictionary
    response={"text": ""}
    while len(response["text"]) < 20 and not response["text"].endswith("}"):
        response = chain(descriptions[i])
        # cut off response text after }
        try:
            response["text"] = response["text"].split("}")[0] + "}"
        except:
            pass
    responses.append(response)
    if (i + 1) % 100 == 0:
        pd.DataFrame(responses).to_json('responses.jsonl', orient='records', lines=True)
        print(f"Reached {i} answers, saving responses to responses.jsonl")
pd.DataFrame(responses).to_json('responses.jsonl', orient='records', lines=True)
        
#     responses.append(chain(descriptions[i]))
#     if i % 100 == 0:
#         pd.DataFrame(responses).to_csv('responses.csv', index=False)
        
# save the responses
# pd.DataFrame(responses).to_csv('responses.csv', index=False)


# %%
tests

# %%
# parse the results
import json

def parse_outputlist(output):
    out_df = list()
    for out in output:
        project_description = out['project_description']
        json_output = json.loads(out['text'])
        try:
            env=json_output['relevant_for_the_environment']
        except:
            env=None
        try:
            reason=json_output['reason']
        except:
            reason=None 
        out_df.append({'project_description':project_description, 'relevant_for_the_environment':env, 'reason':reason})
    return pd.DataFrame(out_df)

# %%
# print df with wrapping text
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)




