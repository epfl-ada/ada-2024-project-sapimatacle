import os
import logging

from tqdm.asyncio import tqdm_asyncio
import aiohttp
import asyncio
import pandas as pd


from src.data.tmdb_tools import get_tmdb_id_from_wikipedia_page_id, get_collection_info

CONCURRENT_REQUESTS = 10

# Example main function to handle multiple page IDs with concurrency control
async def main_tmdb_id(page_ids: int, log_file_name: str):
    # create logs folder
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_file_path = f'../logs/{log_file_name}'
    # check if the same file name exists
    if os.path.exists(log_file_path):
        raise FileExistsError(f"Log file {log_file_path} already exists. Please use a different file name.")
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
async def main_tmdb_info(movie_ids, log_file_name):
    # create logs folder
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_file_path = f'logs/{log_file_name}'
    # check if the same file name exists
    if os.path.exists(log_file_path):
        raise FileExistsError(f"Log file {log_file_path} already exists. Please use a different file name.")
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
        collection_name = [result[0] for result in results]
        collection_id = [result[1] for result in results]
    return collection_name, collection_id

if __name__ == "__main__":
    # Add headers to the DataFrame
    movie_metadata = pd.read_csv('Data/movie.metadata.tsv', sep='\t')
    movie_metadata.columns = [
        'Wikipedia movie ID', 'Freebase movie ID', 'Movie name', 'Movie release date', 
        'Movie box office revenue', 'Movie runtime', 'Movie languages (Freebase ID:name tuples)', 
        'Movie countries (Freebase ID:name tuples)', 'Movie genres (Freebase ID:name tuples)'
    ]
    page_ids = movie_metadata["Wikipedia movie ID"]

    # Extract TMDB IDs for the Wikipedia page IDs
    suffix = 1
    log_file_name = f"tmdb_id_extraction_{suffix}.log"
    while os.path.exists(f'logs/{log_file_name}'):
        suffix += 1
        log_file_name_1 = f"tmdb_id_extraction_{suffix}.log"
    tmdb_ids = asyncio.run(main_tmdb_id(page_ids, log_file_name_1))
    movie_metadata["tmdb_id"] = tmdb_ids

    # Extract information from TMDB for each movie
    # TODO: edit more
    log_file_name_2 = f"tmdb_info_extraction_{suffix}.log"
    while os.path.exists(f'logs/{log_file_name}'):
        suffix += 1
        log_file_name_2 = f"tmdb_info_extraction_{suffix}.log"
    movie_ids = movie_metadata["tmdb_id"]
    collection_name, collection_id = asyncio.run(main_tmdb_info(movie_ids, log_file_name_2))



