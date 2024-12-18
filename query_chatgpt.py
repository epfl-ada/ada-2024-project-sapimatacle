import pdb
import json
import pickle
import asyncio

import httpx
from tqdm.asyncio import tqdm_asyncio
from typing import List

import pandas as pd
from openai import AsyncOpenAI

from src.data.constants import OPENAI_API_KEY
from src.data.utils import get_list_of_characters_per_movie, clean_character_metadata
from src.data.chatgpt_tools import create_prompt, limited_request, str_to_json

def prepare_prompts(plot_df: pd.DataFrame, character_df: pd.DataFrame):
    character_df = clean_character_metadata(character_df, mapping_path='data/ethnicity_mapping.csv')
    char_list_df = get_list_of_characters_per_movie(character_df)
    plot_df_merged = plot_df.merge(char_list_df, on="Wikipedia_movie_ID", how="inner")
    prompts = [create_prompt(characters, plot) for characters, plot in zip(plot_df_merged.Character_name, plot_df_merged.summary_plot)]
    wikipedia_ids = plot_df_merged.Wikipedia_movie_ID
    return wikipedia_ids, prompts

async def query_chargpt(prompts: List[str]):
    httpx_client=httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=10,
            max_keepalive_connections=5
            )
        )
    client = AsyncOpenAI(
        http_client=httpx_client,
        api_key=OPENAI_API_KEY,  # Ensure the API key is set in your environment
    )

    # Use asyncio.gather to handle multiple queries concurrently
    responses = await tqdm_asyncio.gather(*(limited_request(p, client) for p in prompts))

    return responses

if __name__ == '__main__':
    col_names = [
        'Wikipedia_movie_ID', 'Freebase_movie_ID', 'Movie_release_date', 'Character_name', 
        'Actor_date_of_birth', 'Actor_gender', 'Actor_height_m', 'Actor_ethnicity_Freebase_ID', 
        'Actor_name', 'Actor_age_at_movie_release', 'Freebase_character_actor_map_ID', 
        'Freebase_character_ID', 'Freebase_actor_ID'
    ]
    plot_df = pd.read_csv('data/plot_summaries.txt', sep='\t', header=None, names=['Wikipedia_movie_ID', 'summary_plot'])
    character_df = pd.read_csv('data/character.metadata.tsv', names=col_names, sep='\t')
    wikipedia_movie_ids, prompts = prepare_prompts(plot_df, character_df)

    # limit length for testing
    
    ###########################
    # The following lines have to be edited so that we don't reach the usage limit
    prompts = prompts[15000:]
    wikipedia_movie_ids = wikipedia_movie_ids[15000:]
    ###########################

    responses = asyncio.run(query_chargpt(prompts))
    # save responses to file
    with open('data/character_kws/responses_15000-_o4-mini.pkl', 'wb') as f1:
        pickle.dump(responses, f1)
    
    # responses to json
    list_of_json = [str_to_json(r) for r in responses]
    my_dict = {id: json for id, json in zip(wikipedia_movie_ids, list_of_json)}
    json_string = json.dumps(my_dict, indent=4)
    # save dict as json
    with open('data/character_kws/character_kws_15000-_o4-mini.json', 'w') as f2:
        f2.write(json_string)