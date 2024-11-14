import os
import logging
import argparse

from tqdm.asyncio import tqdm_asyncio
import aiohttp
import asyncio
import pandas as pd


from src.data.tmdb_tools import get_tmdb_id_from_wikipedia_page_id, get_collection_info

CONCURRENT_REQUESTS = 10
KEYS = ['collection_name', 'collection_id', "vote_count", "vote_average", "genres", "budget", "revenue", "run_time", "tmdb_origin_country", "tmdb_original_language"]

# Example main function to handle multiple page IDs with concurrency control
async def fetch_tmdb_id(page_ids: int):
    # create logs folder
    if not os.path.exists('logs'):
        os.makedirs('logs')
    # set log file name
    suffix = 1
    log_file_name = f"tmdb_id_extraction_{suffix}.log"
    while os.path.exists(f'logs/{log_file_name}'):
        suffix += 1
        log_file_name = f"tmdb_id_extraction_{suffix}.log"
    log_file_path = f'logs/{log_file_name}'

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(log_file_path)], # File handler to write logs to 'logs/*.log'
                        force=True
                        )
    logger = logging.getLogger(__name__)
    logger.info("Starting the main function ...")
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)  # Limit the number of concurrent requests
    async with aiohttp.ClientSession() as session:
        tasks = [get_tmdb_id_from_wikipedia_page_id(page_id, session, logger=logger, semaphore=semaphore) for page_id in page_ids]
        results = await tqdm_asyncio.gather(*tasks)
        return results
    

# Example usage with delay between requests
async def fetch_tmdb_info(movie_ids):
    # create logs folder
    if not os.path.exists('logs'):
        os.makedirs('logs')
    # set log file name
    suffix = 1
    log_file_name = f"tmdb_info_extraction_{suffix}.log"
    while os.path.exists(f'logs/{log_file_name}'):
        suffix += 1
        log_file_name = f"tmdb_info_extraction_{suffix}.log"
    log_file_path = f'logs/{log_file_name}'

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(log_file_path)], # File handler to write logs to 'logs/*.log'
                        force=True
                        )
    logger = logging.getLogger(__name__)
    logger.info("Starting the main function ...")
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS) 
    async with aiohttp.ClientSession() as session:
        tasks = [get_collection_info(movie_id, session, semaphore, logger) for movie_id in movie_ids]
        results = await tqdm_asyncio.gather(*tasks)
    return results # list of dictionaries

def main(get_id: bool):
    print(get_id)
    if get_id:
        # Add headers to the DataFrame
        movie_metadata = pd.read_csv('Data/movie.metadata.tsv', sep='\t')
        movie_metadata.columns = [
            'Wikipedia movie ID', 'Freebase movie ID', 'Movie name', 'Movie release date', 
            'Movie box office revenue', 'Movie runtime', 'Movie languages (Freebase ID:name tuples)', 
            'Movie countries (Freebase ID:name tuples)', 'Movie genres (Freebase ID:name tuples)'
        ]
        page_ids = movie_metadata["Wikipedia movie ID"]
        # Extract TMDB IDs for the Wikipedia page IDs
        tmdb_ids = asyncio.run(fetch_tmdb_id(page_ids))
        movie_metadata["tmdb_id"] = tmdb_ids
    else:
        movie_metadata = pd.read_csv('Data/movie_metadata_with_tmdb.csv')
        tmdb_ids = movie_metadata["tmdb_id"]
    
    # check if KEYS are in the DataFrame
    if not all(k in movie_metadata.columns for k in KEYS):
        list_of_dict = asyncio.run(fetch_tmdb_info(tmdb_ids))
        for k in KEYS:
            movie_metadata[k] = [d.get(k) for d in list_of_dict]
        movie_metadata.to_csv("Data/movie_metadata_with_tmdb.csv", index=False)
    print("Data fetching complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--get_id", action='store_true', help="Whether to fetch TMDB IDs from Wikipedia page Ids")
    args = parser.parse_args()
    main(args.get_id)
