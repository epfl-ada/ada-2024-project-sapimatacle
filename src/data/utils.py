import time
import pandas as pd
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import Counter
import requests
from bs4 import BeautifulSoup

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
    # Extract the country names from the dictionary strings
    franchise_country = franchise_country.apply(lambda x: ', '.join([country.split(': "')[1].split('"')[0] for country in x.split(', ') if ': "' in country]))
    #region map variable
    region_name_map = {
        'United States of America': 'North America',
        'Canada': 'North America',
        'United Kingdom': 'Europe',
        'Germany': 'Europe',
        'France': 'Europe',
        'Japan': 'Asia',
        'Australia': 'Oceania',
        'Italy': 'Europe',
        'Spain': 'Europe',
        'China': 'Asia',
        'South Korea': 'Asia',
        'India': 'Asia',
        'Sweden': 'Europe',
        'Denmark': 'Europe',
        'Norway': 'Europe',
        'Finland': 'Europe',
        'Netherlands': 'Europe',
        'Belgium': 'Europe',
        'Ireland': 'Europe',
        'New Zealand': 'Oceania',
        'Mexico': 'North America',
        'Brazil': 'South America',
        'Soviet Union': 'Russia',
        'Russia': 'Russia',
        'Hong Kong': 'Asia',
        'Taiwan': 'Asia',
        'Switzerland': 'Europe',
        'Austria': 'Europe',
        'Czech Republic': 'Europe',
        'Poland': 'Europe',
        'Hungary': 'Europe',
        'South Africa': 'Africa',
        'Argentina': 'South America',
        'Chile': 'South America',
        'Peru': 'South America',
        'Colombia': 'South America',
        'Venezuela': 'South America',
        'Portugal': 'Europe',
        'Greece': 'Europe',
        'Turkey': 'Asia',
        'Weimar Republic': 'Europe',
        'Thailand': 'Asia',
        'Philippines': 'Asia',
        'Singapore': 'Asia',
        'German Democratic Republic': 'Europe',
        'Yugoslavia': 'Europe',
        'Czechoslovakia': 'Europe',
        'West Germany': 'Europe',
        'East Germany': 'Europe',
        'Kingdom of Great Britain': 'Europe',
        'Bahamas': 'North America',
        'Ukraine': 'Europe',
        'Cambodia': 'Asia',
        'Romania': 'Europe',
        'Panama': 'North America',
        'Egypt': 'Africa',
        'Morocco': 'Africa',
        'Tunisia': 'Africa',
        'Algeria': 'Africa',
        'Libya': 'Africa',
        'Ethiopia': 'Africa',
        'Indonesia': 'Asia',
        'Malaysia': 'Asia',
        'Iceland': 'Europe',
        'Luxembourg': 'Europe',
        'Sweden': 'Europe',
        'Norway': 'Europe',
        'Finland': 'Europe',
        'Iran': 'Asia',
        'Iraq': 'Asia',
        'Zimbabwe': 'Africa',
        'Slovakia': 'Europe',
        'Serbia': 'Europe',
        'Federal Republic of Yugoslavia': 'Europe',
        'Bulgaria': 'Europe',
        # Add more countries/regions as needed
    }

    # Function to process the countries and map them to regions
    def process_countries(countries, mapping):
        # Split the countries into a list
        country_list = [country.strip() for country in countries.split(',')]
        
        # Count the occurrences of each country
        country_counts = Counter(country_list)
        
        # Sort countries by frequency and remove duplicates
        sorted_countries = [country for country, _ in country_counts.most_common()]
        
        # Map countries to their regions, removing duplicates
        regions = {mapping.get(country, 'Unknown') for country in sorted_countries}
        
        # Return the processed results
        return {
            'countries_sorted': ', '.join(sorted_countries),
            'regions': ', '.join(sorted(regions))
        }
    # Apply the function to the DataFrame
    temp= franchise_country.apply(
        lambda x: pd.Series(process_countries(x, region_name_map))
    )
    franchise_region = temp['regions']
    franchise_country = temp['countries_sorted']
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
        'region': franchise_region.values,
        'average_score': franchise_average_score.values
    }).reset_index(drop=True)
    return franchise_data


col_for_dropna = ['Wikipedia_movie_ID', 'Freebase_movie_ID', 'Movie_release_date',
                  'Actor_gender', 'Actor_name', 'Freebase_character_actor_map_ID',
                  'Freebase_actor_ID']

def clean_character_metadata(data: pd.DataFrame, mapping_path: str, columns: list =col_for_dropna):
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
    time.sleep(1) # To avoid rate limiting
    ethnicity_ids_2 = ethnicity_ids[200:]
    id_to_ethnicity = get_labels_from_freebase_ids(ethnicity_ids_1)
    id_to_ethnicity = id_to_ethnicity | get_labels_from_freebase_ids(ethnicity_ids_2)
    character_df["ethnicity"] = character_df["Actor_ethnicity_Freebase_ID"].map(id_to_ethnicity)

    ethnicity_to_race_dict = pd.read_csv(mapping_path).set_index('Ethnicity')['Group'].to_dict()
    character_df["racial_group"] = character_df.ethnicity.map(ethnicity_to_race_dict)
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


URL ='https://www.minneapolisfed.org/about-us/monetary-policy/inflation-calculator/consumer-price-index-1800-'


def get_inflation_rate(URL): 
    r = requests.get(URL)
    page_body = r.text
    soup = BeautifulSoup(page_body, 'html.parser')
    table = soup.find('table')
    table_rows = table.find_all('tr')
    data = []
    for tr in table_rows:
        td = tr.find_all('td')
        row = [i.text for i in td]
        data.append(row)

    inflation_rate = pd.DataFrame(data, columns=["Year", "CPI","anuel_inflation_rate"])
    # Drop the first row (columns names)
    inflation_rate = inflation_rate.drop(0).reset_index(drop=True)

    # Clean up the data by removing newline characters and extra spaces
    inflation_rate['Year'] = inflation_rate['Year'].str.strip().str.replace('\n', '')
    inflation_rate['CPI'] = inflation_rate['CPI'].str.strip().str.replace('\n', '').astype(float)

    inflation_rate['Year'] = pd.to_numeric(inflation_rate['Year'])
    inflation_rate['CPI'] = pd.to_numeric(inflation_rate['CPI'])

    return inflation_rate


