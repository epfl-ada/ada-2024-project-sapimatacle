import os

import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import backoff
import logging
import asyncio
import aiohttp

from .constants import API_KEY

tqdm.pandas()

@backoff.on_exception(
    backoff.expo,  # Exponential backoff
    aiohttp.ClientResponseError,  # Retry on client response errors
    max_time=5,  # Maximum total wait time (5 seconds)
    giveup=lambda e: e.status != 429,  # Only retry if status code is 429 (Too Many Requests)
    factor=0.2  # Exponential backoff factor
)

async def get_tmdb_id_from_wikipedia_page_id(page_id: int,
                                             session: aiohttp.ClientSession,
                                             semaphore: asyncio.Semaphore,
                                             language: str="en",
                                             logger: logging.Logger|None=None):
    async with semaphore:  # Limit the number of concurrent requests
        # Step 1: Get Wikipedia title from Page ID
        wikipedia_url = f"https://{language}.wikipedia.org/w/api.php"
        wikipedia_params = {
            "action": "query",
            "pageids": page_id,
            "format": "json"
        }
        # Fetch Wikipedia title
        async with session.get(wikipedia_url, params=wikipedia_params) as response:
            data = await response.json()
        
        # Extract title if available
        if "query" in data and "pages" in data["query"]:
            page = data["query"]["pages"].get(str(page_id), {})
            title = page.get("title")
            if title:
                # Step 2: Use the title to retrieve TMDB ID from Wikidata
                wikidata_url = "https://www.wikidata.org/w/api.php"
                wikidata_params = {
                    "action": "wbgetentities",
                    "sites": f"{language}wiki",
                    "titles": title,
                    "props": "claims",
                    "format": "json"
                }
                async with session.get(wikidata_url, params=wikidata_params) as wikidata_response:
                    wikidata_data = await wikidata_response.json()
                    # print(json.dumps(wikidata_data, indent=2))
                    
                    # Extract TMDB ID
                    if "entities" in wikidata_data:
                        for entity in wikidata_data["entities"].values():
                            try:
                                if "P4947" in entity["claims"]:  # P4947 is the TMDB ID property
                                    tmdb_id = entity["claims"]["P4947"][0]["mainsnak"]["datavalue"]["value"]
                                    return tmdb_id
                            except KeyError:
                                if logger:
                                    logger.warning(f"Could not extract TMDB ID from entity: {page_id}")
                                return None
        else:
            if logger:
                logger.warning(f"No title found for page ID: {page_id}")
            return None
            
@backoff.on_exception(
    backoff.expo,  # Exponential backoff
    aiohttp.ClientResponseError,  # Retry on client response errors
    max_time=5,  # Maximum total wait time (5 seconds)
    giveup=lambda e: e.status != 429,  # Only retry if status code is 429 (Too Many Requests)
    factor=0.2
)
async def get_collection_info(movie_id: int, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, logger=logging.Logger):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {'api_key': API_KEY}
    keys = ['collection_name', 'collection_id', "vote_count", "vote_average", "genres", "budget", "revenue", "run_time", "tmdb_origin_country", "tmdb_original_language"]
    my_dict = {k: None for k in keys}

    async with semaphore:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                # get collection info
                if data.get('belongs_to_collection'): # None if it does not belong to a collection
                    collection = data['belongs_to_collection']
                    my_dict['collection_name'] = collection['name']
                    my_dict['collection_id'] = collection['id']
                else:
                    logger.info(msg=f"No collection found for {movie_id}")
                
                # get other info
                my_dict['vote_count'] = data.get('vote_count')
                my_dict['vote_average'] = data.get('vote_average')
                my_dict['genres'] = data.get('genres')
                my_dict['budget'] = data.get('budget')
                my_dict['revenue'] = data.get('revenue')
                my_dict['run_time'] = data.get('runtime')
                my_dict['tmdb_origin_country'] = data.get('origin_country')
                my_dict['tmdb_original_language'] = data.get('original_language')
            else:
                logger.warning(msg=f"Error: Unable to retrieve data for {movie_id}. Status Code: {response.status}")
    return my_dict
    
if __name__ == "__main__":
    # Add headers to the DataFrame
    movie_metadata = pd.read_csv('Data/movie.metadata.tsv', sep='\t')
    movie_metadata.columns = [
        'Wikipedia movie ID', 'Freebase movie ID', 'Movie name', 'Movie release date', 
        'Movie box office revenue', 'Movie runtime', 'Movie languages (Freebase ID:name tuples)', 
        'Movie countries (Freebase ID:name tuples)', 'Movie genres (Freebase ID:name tuples)'
    ]
    page_ids = movie_metadata["Wikipedia movie ID"]
    log_file_name = "tmdb_id_extraction.log"
    tmdb_ids = asyncio.run(main(page_ids, log_file_name))

