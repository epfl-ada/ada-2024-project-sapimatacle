import pandas as pd

def get_franchise_movies(data: pd.DataFrame):
    """Return movies that are part of a franchise and have more than one movie in the franchise.
    Args:
        data: pandas dataframe of 'data/movie_metadata_with_tmdb.csv'

    Returns:
        pd.DataFrame: Franchise movies.
    """
    has_muliple = data.groupby('collection_id').count()['tmdb_id']>1
    valid_idx = has_muliple[has_muliple].index
    data = data[data['collection_id'].isin(valid_idx)].reset_index(drop=True)
    return data