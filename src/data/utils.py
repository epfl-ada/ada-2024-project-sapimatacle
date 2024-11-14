import pandas as pd
import numpy as np
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
    data['Movie release date corrected'] = pd.to_datetime(data['Movie release date'],format='mixed',yearfirst=True)
    data['release_year'] = data['Movie release date corrected'].dt.year
    return data

def get_franchise_data(data: pd.DataFrame):
    """Return franchise data.
    Args:
        data: pandas dataframe of 'franchise_movies_df'

    Returns:
        pd.DataFrame: Franchise data.
    """
    franchise_oldest_release= data.groupby('collection_id')['Movie release date corrected'].min()
    franchise_newest_release= data.groupby('collection_id')['Movie release date corrected'].max()
    franchise_movie_count = data.groupby('collection_id').count()['tmdb_id']
    franchise_length = (franchise_newest_release - franchise_oldest_release).dt.days
    franchise_length_years = (franchise_length / 365).round(0)
    franchise_revenue = data.groupby('collection_id').apply(
        lambda x: np.nan if x[['Movie box office revenue', 'revenue']].isnull().any().any() else (
            x['Movie box office revenue'].sum() if x['Movie box office revenue'].notnull().all() else (
                x['revenue'].sum() if x['revenue'].notnull().all() else 0
            )
        )
    ).astype(float)
    franchise_country = data.groupby('collection_id')['Movie countries (Freebase ID:name tuples)'].apply(lambda x: ', '.join(x.unique()))
    franchise_average_score= data.groupby('collection_id')['vote_average'].mean()
    franchise_data = pd.DataFrame({
        'collection_id': franchise_oldest_release.index,
        'collection_name': data.groupby('collection_id')['collection_name'].first(),
        'oldest_release': franchise_oldest_release.values,
        'newest_release': franchise_newest_release.values,
        'movie_count': franchise_movie_count.values,
        'franchise_length': franchise_length,
        'franchise_length_years': franchise_length_years,
        'revenue': franchise_revenue.values,
        'country': franchise_country.values,
        'average_score': franchise_average_score.values
    }).reset_index(drop=True)
    return franchise_data
