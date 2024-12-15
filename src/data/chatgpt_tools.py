import json
from typing import List
from asyncio import Semaphore

import backoff
import openai
import pandas as pd
from openai import AsyncOpenAI

from .constants import OPENAI_API_KEY
from .utils import get_franchise_movies, clean_character_metadata, custom_autopct, get_labels_from_freebase_ids

INSTRUCTION = """
Given a list of character names and a movie plot summary, return a JSON object where each character name is a key,
and the value is a list of adjectives that describe the character. Do not repeat the same words in the list.
If a character is not mentioned or described in the plot, return an empty list for that character.
The output should be directly loaded by json.loads() in Python.
"""

SEMAPHORE = Semaphore(5)

def str_to_json(json_string: str):
    start = json_string.find('{')
    end = json_string.rfind('}') + 1
    try:
        json_obj = json.loads(json_string[start:end])
    except json.JSONDecodeError:
        print("Error: JSON string could not be decoded.")
        json_obj = {}
    return json_obj

def create_prompt(characters: List[str], plot: str):
    prompt = f"""
    Character names: {characters}
    Movie plot summary: {plot}
    """
    return prompt

# Define an async function to handle independent queries
async def get_answer(prompt: str, client: AsyncOpenAI) -> str:
    # Call the API with an independent query
    response = await client.chat.completions.create(
        messages=[
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        model="gpt-4o-mini"
        #model="gpt-3.5-turbo-0125"
        #model="gpt-3.5-turbo-1106"
    )
    # Extract and return the assistant's reply
    return response.choices[0].message.content

@backoff.on_exception(
    backoff.expo,  # Exponential backoff
    openai.RateLimitError,  # Retry on client response errors
    max_time=10,  # Maximum total wait time (5 seconds)
    factor=0.2  # Exponential backoff factor
)
async def limited_request(prompt, client):
    async with SEMAPHORE:
        return await get_answer(prompt, client)