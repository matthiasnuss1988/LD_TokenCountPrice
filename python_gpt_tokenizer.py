import os
import re
import openai
import tiktoken
import textwrap
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decrypt import load_key

# Set up OpenAI API key
openai.api_key = load_key("encrypted_api_key.bin")
messages = [
        {"role": "system", "content": ""}
            ]       



def get_model_params(model_name):
    #encoding = ''
    #token_limit = 0
    max_tokens = None
    encoding = None
    pricing = None
    if 'gpt-3.5' in model_name:
        encoding = 'cl100k'
        token_limit = 4096
    elif any(word in model_name for word in ['similarity', 'embedding', 'search', 'query']):
        encoding = 'r50k'
        if '001' in model_name:
            token_limit = 2048
        elif '002' in model_name:
            token_limit = 8191
            encoding = 'cl100k'
        else:
            token_limit = 2048
            encoding = 'r50k'
    elif 'code' in model_name and any(word in model_name for word in ['similarity', 'embedding', 'search', 'query']):
        encoding = 'r50k'
        if '001' in model_name:
            token_limit = 2046
        elif '002' in model_name:
            token_limit = 8191
            encoding = 'cl100k'
        else:
            token_limit = 2046
            encoding = 'r50k'
    elif 'code' in model_name:
        encoding = 'p50k'
        if '001' in model_name:
            token_limit = 2046
        elif '002' in model_name:
            token_limit = 8000
            encoding = 'p50k'
        else:
            token_limit = 8000
            encoding = 'cl100k'
    elif 'davinci' in model_name:
        encoding = 'r50k'
        if '003' in model_name or '002' in model_name:
            token_limit = 4000
            encoding = 'p50k'
        elif 'codex' in model_name:
            token_limit = 4096
        elif 'edit' in model_name:
            encoding = 'r50k'
        else:
            token_limit = 2048
    else:
        token_limit = 2048
        encoding = 'r50k'
    if 'edit' in model_name:
        encoding += '_edit'
    else:
        encoding += '_base'
    token_price=get_model_price(model_name)
    params = {'model_name': model_name,'encoding': encoding, 'token_limit': token_limit, 'token_price': token_price}
    return params

def get_model_price(model_name):
    if 'gpt-3.5' in model_name:
        return 0.002 / 1000
    if 'ada' in model_name:
        return 0.0004 / 1000
    elif 'babbage' in model_name:
        return 0.0005 / 1000
    elif 'curie' in model_name:
        return 0.0020 / 1000
    elif 'davinci' in model_name:
        return 0.0200 / 1000
    elif 'fine_tuning' and 'ada' in model_name:
        return 0.0016 / 1000
    elif 'fine_tuning'  and 'babbage' in model_name:
        return 0.0024 / 1000
    elif 'fine_tuning'  and 'curie' in model_name:
        return 0.0120 / 1000
    elif 'fine_tuning'  and 'davinci' in model_name:
        return 0.1200 / 1000
    elif any(word in model_name for word in ['embedding', 'query', 'search', 'similarity']):
        if 'ada' in model_name:
            return 0.0004 / 1000
    return None

def compare_encodings():
    models = openai.Model.list()
    encodings =[]
    for model in models['data']:
        print(get_model_params(model.id))
        encoding_calculated = get_model_params(model.id)['encoding']
        try:
            encoding_tiktoken = str(tiktoken.encoding_for_model(model.id)).split(' ')[1][1:-2]
            if encoding_tiktoken  == encoding_calculated:
                match='yes'
            else:
                match= 'no'
            #print({'Model ID': model.id, 'Match': match})
        except KeyError:
            encoding_tiktoken= 'undefined'
        #print({'Model ID': model.id, 'Encoding': encoding_tiktoken})
        #print({'Model ID': model.id, 'Encoding': encoding_calculated})
    return None

#compare_encodings()





# Define function to count number of tokens in text
def count_tokenprice_tiktoken(text, model_name):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding(get_model_params(model_name)['encoding'])
    num_tokens = 0
    #encodingclean = str(encoding).split(' ')[1][1:-2]
    try:
        if 'gpt-3.5-turbo' in model_name:
            messages[0]["content"] = text
            for message in messages:
                num_tokens += 4
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += -1
            num_tokens += 2
            tiktoken_price_estimate=get_model_price(model_name)*num_tokens
        else:
            num_tokens += len(encoding.encode(text))
            tiktoken_price_estimate=get_model_price(model_name)*num_tokens
        params = {'tiktoken_token_count': num_tokens,'tiktoken_price_estimate': tiktoken_price_estimate, 'encoding': encoding}
        return params
    except:
        print({'Model ID': model_name, 'Error': 'Tokens could not be calculated'})
        params = {'tiktoken_token_count': None, 'tiktoken_price_estimate': None, 'encoding': encoding}
        return params

def count_tokens_GPT35(messages, model_engine):
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=messages,
        temperature=0.1,
    )
    token_count_openai=response["usage"]["prompt_tokens"]
    return token_count_openai

def count_tokens_GPT3(text, model_name):
    response = openai.Completion.create(
        engine=model_name,
        prompt=text,
        temperature=0.2,
    )
    token_count_openai=response["usage"]["prompt_tokens"]
    return token_count_openai

def count_tokens_embedding(text, model_name):
   response = openai.Embedding.create(
       input = text,
       model=model_name
    )
   token_count_openai=response['usage']['prompt_tokens']
   return token_count_openai

def chose_openai_tokencounter(text, model_name):
    if any(word in model_name for word in ['similarity', 'embedding', 'search', 'query']):
        token_counts_openai=count_tokens_embedding(text, model_name)
    elif 'gpt-3.5-turbo' in model_name:
        messages[0]["content"] = text
        token_counts_openai=count_tokens_GPT35(messages, model_name)
    else: 
        token_counts_openai=count_tokens_GPT3(text, model_name)
    return token_counts_openai
    
def compare_modelbased_tokencounts():
    models = openai.Model.list()
    folder_path = "./chunks"
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    with open(os.path.join(folder_path, files[0]), mode="r", encoding="latin1") as f:
        text = f.read()
        for model in models['data']:
            if all(word not in model['id'] for word in ["whisper","audio", "2020"]):
                params = count_tokenprice_tiktoken(text, model['id'])
                token_counts_tiktoken = params["tiktoken_token_count"]
                encoding = params["encoding"]
                tokens_price = params["tiktoken_price_estimate"]
                print({'Model ID': model['id'], 'Encoding': encoding, 'Token tiktoken': token_counts_tiktoken, 'Tokens price': tokens_price})

compare_modelbased_tokencounts()




# files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
# # Count tokens in each file and print to console
# for filename in files:
#     with open(os.path.join(folder_path, filename), mode="r", encoding="latin1") as f:
#         text = f.read()
#         messages[0]["content"] = text
#         #Count tokens tiktoken
#         token_counts_tiktoken=count_tokens_tiktoken(messages,model_engine)
#         #Count tokens openai
#         if model_engine=="gpt-3.5-turbo-0301":
#             token_counts_openai=count_tokens_chat(messages,model_engine)
#         else:
#             token_counts_openai=count_tokens_embedding(messages,model_engine)
#         print(f"{filename},{token_counts},{token_counts_tiktoken}")
