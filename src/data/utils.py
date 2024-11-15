import time
import pandas as pd
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON

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


col_for_dropna = ['Wikipedia_movie_ID', 'Freebase_movie_ID', 'Movie_release_date',
                  'Actor_gender', 'Actor_name', 'Freebase_character_actor_map_ID',
                  'Freebase_actor_ID']

def clean_character_metadata(data: pd.DataFrame, columns: list =col_for_dropna):
    """Drop rows if specified columns have missing values. Also add 
    Args:
        data: pandas dataframe of 'data/character.metadata.tsv'
        columns: list of columns to check for missing values.

    Returns:
        pd.DataFrame: Cleaned character metadata.
    """
    character_df = data.dropna(subset=columns).reset_index(drop=True)
    print(f"Number of rows dropped: {data.shape[0] - character_df.shape[0]}/{data.shape[0]}")
    print(f"{character_df.shape[0]} rows remaining.")
    ethnicity_ids = character_df["Actor_ethnicity_Freebase_ID"].dropna().unique().tolist()
    ethnicity_ids_1 = ethnicity_ids[:200] # The header length is limited, so divide into two parts
    time.sleep(0.1) # To avoid rate limiting
    ethnicity_ids_2 = ethnicity_ids[200:]
    id_to_ethnicity = get_labels_from_freebase_ids(ethnicity_ids_1)
    id_to_ethnicity = id_to_ethnicity | get_labels_from_freebase_ids(ethnicity_ids_2)
    character_df["ethnicity"] = character_df["Actor_ethnicity_Freebase_ID"].map(id_to_ethnicity)
    return character_df

def custom_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.1f}%\n({v:d})'.format(p=pct,v=val)
    return my_autopct

def get_labels_from_freebase_ids(freebase_ids):
    # Initialize SPARQL wrapper for the Wikidata endpoint
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    
    # Convert Freebase IDs list into a string format for SPARQL
    freebase_values = " ".join([f'"{id_}"' for id_ in freebase_ids])
    
    # SPARQL query to get Wikidata labels by Freebase IDs
    query = f"""
    SELECT ?freebase_id ?label WHERE {{
      VALUES ?freebase_id {{ {freebase_values} }}
      ?item wdt:P646 ?freebase_id;
            rdfs:label ?label.
      FILTER(LANG(?label) = "en")
    }}
    """
    
    # Set up the SPARQL query
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    # Execute the query and retrieve the results
    results = sparql.query().convert()
    
    # Parse the results into a dictionary
    labels = {result["freebase_id"]["value"]: result["label"]["value"] for result in results["results"]["bindings"]}
    
    return labels

