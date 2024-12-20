import time
import pandas as pd
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import Counter
import requests
from bs4 import BeautifulSoup
import ast
from sklearn.feature_extraction.text import CountVectorizer


from .sentiment_analysis import get_kw_dataframe

TMDB_GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "Thriller", "TV Movie", "War", "Western"]

def clean_categories(category_string):
    # Remove brackets and quotes
    cleaned_category = category_string.replace("[", "").replace("]", "").replace("'", "").split(", ")
    # Return category list
    return cleaned_category

def get_inflation_rate(): 
    URL ='https://www.minneapolisfed.org/about-us/monetary-policy/inflation-calculator/consumer-price-index-1800-'
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

def extract_genres(genres_str):
    try:
        genres_list = ast.literal_eval(genres_str)
        return [genre['name'] for genre in genres_list]
    except (ValueError, SyntaxError):
        return []

def get_franchise_movies(data: pd.DataFrame, data_2: pd.DataFrame, path_missingdates: str):
    """Return movies that are part of a franchise and have more than one movie in the franchise.
    Args:
        data: pandas dataframe of 'data/movie_metadata_with_tmdb.csv'
        data_2: pandas dataframe of the inflation rate from the web sit of the Federal Reserve Bank of Minneapolis
    Returns:
        pd.DataFrame: Franchise movies.
    """
    # Open the missing_dates_manualsearch.csv file
    missing_dates = pd.read_csv(path_missingdates)
    missing_dates.dropna(subset=['Movie release date'], inplace=True)
    missing_dates.dtypes
    missing_dates=missing_dates.astype({'Movie release date': 'int'})
    missing_dates=missing_dates.astype({'Movie release date': 'str'})

    # Merge the missing_dates with the data
    data = pd.merge(data,missing_dates[['Wikipedia movie ID','Movie release date']],on='Wikipedia movie ID',how='outer',suffixes=('','_y'))
    data['Movie release date'] = data['Movie release date'].combine_first(data['Movie release date_y'])
    data.drop(columns=['Movie release date_y'],inplace=True)


    # Only take the movies that have a collection id, and where there is more than one movie in the collection
    has_muliple = data.groupby('collection_id').count()['tmdb_id']>1
    valid_idx = has_muliple[has_muliple].index
    data = data[data['collection_id'].isin(valid_idx)].reset_index(drop=True)

    # Drop the whole franchise where a movie has no release date
    data = data.groupby('collection_id').filter(lambda x: x['Movie release date'].notna().all())
    
    # Correct the release date
    data['Movie release date corrected'] = pd.to_datetime(data['Movie release date'],format='mixed',yearfirst=True)

    # add a column with the release year
    data['release_year'] = data['Movie release date corrected'].dt.year

    # add a column with the numerotation of the movies in the collection by release date order
    data['movie_order'] = data.groupby('collection_id')['Movie release date corrected'].rank(method='first')

    # box office merge and drop the original columns
    data['box_office']=data['Movie box office revenue'].apply(lambda x: x if x!=np.nan else data['revenue'])
    data.drop(columns=['Movie box office revenue','revenue'],inplace=True)

    # add a profit column
    data['profit'] = data['box_office'] - data['budget']

    # add a column withe the years between the movies of the same collection
    data["years_diff_bt_pre_movies"] = (
        data.groupby("collection_id", group_keys=False)
        .apply(lambda group: group.sort_values("movie_order"))
        .groupby("collection_id")["release_year"]
        .diff()
    )

    # replace the 0 values with nan for the buget column
    data['budget'] = data['budget'].apply(lambda x: np.nan if x==0 else x)

    #tacking into account inflation for revenue and budget
    data['CPI'] = data.merge(data_2[['Year', 'CPI']], how='left', left_on='release_year', right_on='Year')['CPI']
    base_year_cpi= data_2.loc[data_2['Year'] == 2024, 'CPI'].iloc[0] #base year 2024
    #Real Price = Nominal Price (at the time) × CPI in Base Year / CPI in Year of Price
    data['real_revenue']= data['box_office']*base_year_cpi/data['CPI'].iloc[0]
    data['real_budget']= data['budget']*base_year_cpi/data['CPI'].iloc[0]
    data['real_profit']= data['real_revenue'] - data['real_budget']
    #Ratio revenue over budget : 
    data['ratio_revenue_budget']= data['real_revenue']/data['real_budget']

    # Clean the genres 
    data['genres'] = data['genres'].apply(extract_genres)
    # Add the number of genres for each movie
    data['num_genres'] = data['genres'].apply(len)
    
    # Add a column with the number of movies in the collection
    collection_counts = data['collection_id'].value_counts()
    data['collection_size'] = data['collection_id'].map(collection_counts) # Map the counts back to the original DataFrame

    return data

def get_1_2_movies(data): 
    """
    Function to get the 1_st and 2_nd movie franhcise to analyse it in the tree and KNN model
    """
    # Drop unnecessary columns
    data=data.drop(columns=[
    'Freebase movie ID', 'Movie release date', 'Movie genres (Freebase ID:name tuples)', 'Movie languages (Freebase ID:name tuples)', 
    'tmdb_original_language', 'Movie release date corrected', 'box_office', 
    'Movie runtime', 'Movie countries (Freebase ID:name tuples)', 'CPI', 'budget', 'tmdb_id','profit'
    ])

    data['movie_order'] = data['movie_order'].astype(int)

    data = data[(data['movie_order'] == 1) | (data['movie_order'] == 2)]
    data = data.pivot(index='collection_id', columns='movie_order')
    data.columns = ['_'.join(map(str, col)).strip() for col in data.columns]
    data = data.drop(columns =['years_diff_bt_pre_movies_1','collection_size_2'], axis=1)
    data = data.reset_index()

    # genre vecotrization
    data['genres_1'] = data['genres_1'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(", "),token_pattern=None)
    genre_matrix = vectorizer.fit_transform(data['genres_1'])
    genre_df = pd.DataFrame(genre_matrix.toarray(), columns=[f'genre_1_{col}' for col in vectorizer.get_feature_names_out()])
    data = pd.concat([data, genre_df], axis=1)

    data['genres_2'] = data['genres_2'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(", "),token_pattern=None)
    genre_matrix = vectorizer.fit_transform(data['genres_2'])
    genre_df = pd.DataFrame(genre_matrix.toarray(), columns=[f'genre_2_{col}' for col in vectorizer.get_feature_names_out()])
    data = pd.concat([data, genre_df], axis=1)

    data['tmdb_origin_country_1'] = data['tmdb_origin_country_1'].apply(lambda x: x.replace("[", "").replace("]", "").replace("'", ""))
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(", "),token_pattern=None)
    country_matrix = vectorizer.fit_transform(data['tmdb_origin_country_1'])
    country_df = pd.DataFrame(country_matrix.toarray(), columns=[f'tmdb_origin_country_1_{col}' for col in vectorizer.get_feature_names_out()])
    data = pd.concat([data, country_df], axis=1)

    data['tmdb_origin_country_2'] = data['tmdb_origin_country_2'].apply(lambda x: x.replace("[", "").replace("]", "").replace("'", ""))
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(", "),token_pattern=None)
    country_matrix = vectorizer.fit_transform(data['tmdb_origin_country_2'])
    country_df = pd.DataFrame(country_matrix.toarray(), columns=[f'tmdb_origin_country_2_{col}' for col in vectorizer.get_feature_names_out()])
    data = pd.concat([data, country_df], axis=1)

    data['same_genres'] = (data['genres_1'] == data['genres_2']).astype(int)
    return data 
    
def get_movie(data: pd.DataFrame, data_2: pd.DataFrame):
    """Return movies with the same features as the franchise movies. 
    Args:
        data: pandas dataframe of 'data/movie_metadata_with_tmdb.csv'
        data_2: pandas dataframe of the inflation rate from the web sit of the Federal Reserve Bank of Minneapolis
    Returns:
        pd.DataFrame: Franchise movies.
    """
    data = data.copy(deep=True)
    # Correct the release date
    data['Movie release date corrected'] = pd.to_datetime(data['Movie release date'],format='mixed',yearfirst=True, errors='coerce')

    # add a column with the release year
    data['release_year'] = data['Movie release date corrected'].dt.year

    # box office merge and drop the original columns
    data['box_office']=data['Movie box office revenue'].apply(lambda x: x if x!=np.nan else data['revenue'])
    data.drop(columns=['Movie box office revenue','revenue'],inplace=True)

    # add a profit column
    data['profit'] = data['box_office'] - data['budget']

    # replace the 0 values with nan for the buget column
    data['budget'] = data['budget'].apply(lambda x: np.nan if x==0 else x)

    #tacking into account inflation for revenue and budget
    data['CPI'] = data.merge(data_2[['Year', 'CPI']], how='left', left_on='release_year', right_on='Year')['CPI']
    base_year_cpi= data_2.loc[data_2['Year'] == 2024, 'CPI'].iloc[0] #base year 2024
    #Real Price = Nominal Price (at the time) × CPI in Base Year / CPI in Year of Price
    data['real_revenue']= data['box_office']*base_year_cpi/data['CPI'].iloc[0]
    data['real_budget']= data['budget']*base_year_cpi/data['CPI'].iloc[0]
    data['real_profit']= data['box_office'] - data['budget']

    # Clean the genres 
    data['genres'] = data['genres'].apply(extract_genres)
    return data

def get_clean_franchise_movies(data: pd.DataFrame):
    col_for_dropna =['profit','real_budget','real_revenue','real_profit','CPI','release_year','movie_order','genres']

def get_franchise_data(data: pd.DataFrame):
    """Return franchise data.
    Args:
        data: pandas dataframe of 'franchise_movies_df'

    Returns:
        pd.DataFrame: Franchise data.
    """
    data = data.copy()
    franchise_oldest_release= data.groupby('collection_id')['Movie release date corrected'].min()
    franchise_newest_release= data.groupby('collection_id')['Movie release date corrected'].max()
    franchise_movie_count = data.groupby('collection_id').count()['tmdb_id']
    franchise_length = (franchise_newest_release - franchise_oldest_release).dt.days
    franchise_length_years = (franchise_length / 365).round(0)
    franchise_average_years_bt_movies = franchise_length_years / (franchise_movie_count-1)
    franchise_revenue_avg = data.groupby('collection_id')['real_revenue'].mean()
    franchise_genre = data.groupby('collection_id')['genres'].apply(lambda x: ', '.join(set([genre for sublist in x for genre in sublist])))
    franchise_country = data.groupby('collection_id')['Movie countries (Freebase ID:name tuples)'].apply(lambda x: ', '.join(x.unique()))
    franchise_budget_avg = data.groupby('collection_id')['real_budget'].mean()
    franchise_runtine_avg = data.groupby('collection_id')['Movie runtime'].mean()
    franchise_ratio_rb = franchise_revenue_avg/franchise_budget_avg


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
        'genres': franchise_genre.values,
        'oldest_release': franchise_oldest_release.values,
        'newest_release': franchise_newest_release.values,
        'runtime_avg': franchise_runtine_avg.values,
        'movie_count': franchise_movie_count.values,
        'average_years_bt_movies': franchise_average_years_bt_movies.values,
        'franchise_length': franchise_length,
        'franchise_length_years': franchise_length_years,
        'revenue_avg': franchise_revenue_avg.values,
        'budget_avg': franchise_budget_avg.values,
        'ratio_rb': franchise_ratio_rb.values,
        'country': franchise_country.values,
        'region': franchise_region.values,
        'average_score': franchise_average_score.values
    }).reset_index(drop=True)
    return franchise_data


col_for_dropna = ['Wikipedia_movie_ID', 'Freebase_movie_ID', 'Movie_release_date',
                  'Actor_gender', 'Actor_name', 'Freebase_character_actor_map_ID',
                  'Freebase_actor_ID']

def clean_character_metadata(data: pd.DataFrame, mapping_path: str, columns: list =col_for_dropna):
    """Cleans the character metadata by dropping rows with missing values in specified columns,
    mapping ethnicity IDs to their labels, and mapping ethnicities to racial groups.

    Args:
        data (pd.DataFrame): DataFrame containing character metadata 'data/character.metadata.tsv'.
        mapping_path (str): Path to the CSV file containing the mapping of ethnicities to racial groups.
        columns (list): List of columns to check for missing values. Defaults to
        ['Wikipedia_movie_ID', 'Freebase_movie_ID', 'Movie_release_date', 'Actor_gender',
        'Actor_name', 'Freebase_character_actor_map_ID', 'Freebase_actor_ID'].

    Returns:
        pd.DataFrame: Cleaned character metadata.
    """
    print(f"Dropping character data rows with missing values in any of {col_for_dropna}.")
    character_df = data.dropna(subset=columns).reset_index(drop=True)
    print(f"Number of rows dropped: {data.shape[0] - character_df.shape[0]}")
    print(f"{character_df.shape[0]} rows remaining.")

    # Get etbnicity related columns
    ethnicity_ids = character_df["Actor_ethnicity_Freebase_ID"].dropna().unique().tolist()
    ethnicity_ids_1 = ethnicity_ids[:200] # The header length is limited, so divide into two parts
    time.sleep(1) # To avoid rate limiting
    ethnicity_ids_2 = ethnicity_ids[200:]
    id_to_ethnicity = get_labels_from_freebase_ids(ethnicity_ids_1)
    id_to_ethnicity = id_to_ethnicity | get_labels_from_freebase_ids(ethnicity_ids_2)
    character_df["ethnicity"] = character_df["Actor_ethnicity_Freebase_ID"].map(id_to_ethnicity)
    ethnicity_to_race_dict = pd.read_csv(mapping_path).set_index('Ethnicity')['Group'].to_dict()
    character_df["racial_group"] = character_df.ethnicity.map(ethnicity_to_race_dict)
    kw_df = get_kw_dataframe('data/character_kws')
    character_df["char_name_lower"] = character_df["Character_name"].str.lower()
    merged_df = character_df.merge(kw_df, on=["Wikipedia_movie_ID", "char_name_lower"], how="left")
    return merged_df

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

def get_genre_counts(data = pd.DataFrame):
    genre_counts = Counter()

    for genres in data['Movie genres (Freebase ID:name tuples)']:
        if pd.notnull(genres):
            # Split the genres by comma and strip any extra spaces
            genre_list = [genre.split(":")[1].strip().replace("}", "") for genre in genres.split(",") if ":" in genre]
            genre_counts.update(genre_list)
    
    genre_counts_df = pd.DataFrame.from_dict(genre_counts, orient='index', columns=['counts']).reset_index()
    genre_counts_df.columns = ['genre', 'counts']
    genre_counts_df['genre']=genre_counts_df['genre'].str.replace('}','')
    genre_counts_df = genre_counts_df.sort_values(by='counts', ascending=False)
    # Clean up genre names by removing double quotes
    genre_counts_df['genre'] = genre_counts_df['genre'].str.replace('"', '')

    # Calculate the total count of all genres
    total_genres_count = genre_counts_df['counts'].sum()

    return total_genres_count, genre_counts_df

def get_list_of_characters_per_movie(character_df: pd.DataFrame):
    """Get a list of characters per movie from the given DataFrame. Drop characters with no ethnicity entry.

        character_df (pd.DataFrame): DataFrame containing character information, 
                                     including columns 'Wikipedia_movie_ID', 'Character_name', and 'ethnicity'.

        pd.DataFrame: DataFrame with 'Wikipedia_movie_ID' and a list of 'Character_name' for each movie, 
                      excluding movies with no character names.
    """
    # Drop rows without ethnicity entry
    char_eth_df = character_df.dropna(subset="ethnicity")
    # Get list of characters per movie, excluding NaN values
    char_eth_df = char_eth_df.groupby("Wikipedia_movie_ID", as_index=False)["Character_name"].apply(lambda x: x.dropna().tolist())
    return char_eth_df[char_eth_df.Character_name.str.len() > 0].reset_index(drop=True)

def create_is_from_asia(df):
    """Create a boolean columns indicating if the movie is from Asia

    Args:
        df: movies_df, franchise_df, or movies_no_franchise_df
    Return:
        df: dataframe with a new column `is_from_asia`
    """
    df = df.copy(deep=True)
    asian_countries = [
    # Asia
    "AF", "BD", "BT", "BN", "KH", "CN", "TL", "IN", "ID", "JP", 
    "KZ", "KG", "LA", "MY", "MV", "MN", "MM", "NP", "PK", "PH", 
    "SG", "KR", "LK", "TJ", "TH", "TM", "UZ", "VN", "KP",
    # Russia
    "RU",
    # Oceania
    "AS", "AU", "CK", "FJ", "FM", "GU", "KI", "MH", "NR", "NC", 
    "NZ", "NU", "PW", "PG", "PN", "SB", "TK", "TO", "TV", "VU", "WF", "WS"
]
    list_of_countries = df.tmdb_origin_country.fillna('')
    for i in range(len(list_of_countries)):
        list_of_countries.iloc[i] = list_of_countries.iloc[i].strip("[]").replace("'", "").split(", ")

    # check if asia is in the list of countries
    assert len(df) == len(list_of_countries), "Length of dataframe and list of countries do not match"
    bool_list = []
    for l in list_of_countries:
        if any([country in asian_countries for country in l]):
            bool_list.append(True)
        else:
            bool_list.append(False)
    df["is_from_asia"] = bool_list
    return df

def create_ethnicity_columns(movie_df, character_df):
    """Create new columns in the movie dataframe with the number of racial groups and female_ratio in the movie.
    
    Args:
        movie_df (pd.DataFrame): DataFrame containing movie data with at least a 'Wikipedia movie ID' column.
        character_df (pd.DataFrame): DataFrame containing character data with 'Wikipedia_movie_ID' and 'racial_group' columns.

    Returns:
        pd.DataFrame: The original movie dataframe with additional columns for each racial group
        (e.g., 'num_Asian', 'num_Black') and female_ratio. The count represents the number of characters from each racial group.
    """
    movie_df = movie_df.copy(deep=True)
    def str_to_list(x):
        my_list = ",".join(x).strip(",").split(",")
        my_list = list(filter(lambda x: x != "", my_list))
        return my_list

    character_df[["Wikipedia_movie_ID", 'racial_group']].fillna("").groupby("Wikipedia_movie_ID")["racial_group"].apply(str_to_list)
    character_df_unique_racial_group = character_df[["Wikipedia_movie_ID", 'racial_group']].fillna("").groupby("Wikipedia_movie_ID")["racial_group"].apply(str_to_list).explode().reset_index()
    racial_group_counts = character_df_unique_racial_group.groupby(['Wikipedia_movie_ID', 'racial_group']).size().unstack(fill_value=0).reset_index()
    racial_group_counts.columns.name = None

    # Add "num_" prefix to racial group column names
    racial_group_counts = racial_group_counts.rename(columns=lambda x: f'num_{x}' if x != 'Wikipedia_movie_ID' else x)
    movie_df = movie_df.merge(racial_group_counts, left_on="Wikipedia movie ID", right_on='Wikipedia_movie_ID', how='left')
    movie_df = movie_df.drop(columns=['Wikipedia_movie_ID'])

    ## Add gender ratio
    gender_df = character_df[["Wikipedia_movie_ID", "Actor_gender"]].copy()
    gender_df.loc[:, "is_woman"] = (gender_df["Actor_gender"] == "F").astype(int)
    female_ratio = gender_df.groupby("Wikipedia_movie_ID")["is_woman"].mean().rename('female_ratio')
    movie_df = movie_df.merge(female_ratio, left_on="Wikipedia movie ID", right_on="Wikipedia_movie_ID", how="left")

    return movie_df

def get_dummies_for_tree(movies_df):
    """Create one-hot encoded columns for 'tmdb_origin_country', 'tmdb_original_language', and 'genres'.
    This function processes the provided DataFrame by generating binary indicator columns for each unique country in
    'tmdb_origin_country', each unique language in 'tmdb_original_language', and each genre in 'genres'. The original
    columns are dropped from the DataFrame after encoding. Some columns such as IDs were removed.

    Args:
        movies_df (pd.DataFrame): The DataFrame containing movie data with at least the columns 'tmdb_origin_country',
                                  'tmdb_original_language', and 'genres'.
    Returns:
        pd.DataFrame: A new DataFrame with one-hot encoded columns for countries, languages, and genres,
                      with the original columns removed.
    """
    # only keep the columns we need
    col_for_tree  = ['vote_count',
                    'vote_average', 'genres','run_time', 'tmdb_origin_country', 'tmdb_original_language',
                    'num_Asian', 'num_Black', 'num_Hispanic', 'num_Middle Eastern',
                    'num_Native American', 'num_Others', 'num_White',
                    'release_year', 'real_revenue', 'real_budget', 'female_ratio']
    
    data = movies_df.copy(deep=True)
    # Preliminary: only keep the columns we need
    data = data[col_for_tree]

    # STEP 1: create binary "from" columns
    data["tmdb_origin_country"] = data["tmdb_origin_country"]
    list_of_countries =data["tmdb_origin_country"].str.strip("[]").str.replace("'", "").str.split(", ")

    # Flatten the list of countries and count occurrences
    all_countries = list_of_countries.dropna().explode().unique()
    all_countries = filter(lambda x: x != "", all_countries) # drop empty strings

    country_df = pd.DataFrame({f"from_{country}": data["tmdb_origin_country"].apply(
        lambda x: (country in x) if isinstance(x, str) else False) for country in all_countries})
    data = pd.concat([data, country_df], axis=1)

    # STEP 2: create binary "lang" columns
    lang_df = pd.get_dummies(data.tmdb_original_language, prefix='lang').astype(int)
    data = pd.concat([data, lang_df], axis=1)

    # STEP 3: create binary "genre" columns
    genre_df = pd.DataFrame({f"is_{genre}": data["genres"].apply(
        lambda x: (genre in x) if isinstance(x, list) else False) for genre in TMDB_GENRES})
    data = pd.concat([data, genre_df], axis=1)

    # STEP 4: drop the original columns
    data.drop(columns=["tmdb_origin_country", "tmdb_original_language", "genres"], inplace=True)
    return data

def get_tree_df(franchise_df, movies_no_franchise_df, random_state=42):
    """Create a DataFrame for tree-based models. The function generates one-hot encoded columns for countries, languages,
    while removing some columns unnecessary for model training. The dataframe containing all the 1st movies in the franchise that has
    at least one subsequent movie and non-franchise is returned. Non-franchise data is subsampled to match the size of the franchise data.

    Args:
        franchise_df: generated by get_franchise_movies
        movies_no_franchise_df: dataframe of movies without franchise
        random_state: random seed for reproducibility

    Returns:
        pd.DataFrame
    """
    # only preserve collection_size > 1
    fr_tree_df = franchise_df[franchise_df["collection_size"] > 1]
    # only pick the first movie in each collection
    fr_tree_df= fr_tree_df[fr_tree_df["movie_order"] == 1]
    fr_tree_df = get_dummies_for_tree(fr_tree_df)
    fr_tree_df["in_franchise"] = 1
    non_fr_tree_df = get_dummies_for_tree(movies_no_franchise_df)
    non_fr_tree_df["in_franchise"] = 0
    fr_size = len(fr_tree_df)
    # subsample non-franchise data to match the size of the franchise data
    non_fr_tree_df = non_fr_tree_df.sample(n=fr_size, random_state=random_state)
    tree_df = pd.concat([fr_tree_df, non_fr_tree_df], axis=0).copy()
    # fillna for from_{country} and lang_{lang} columns
    # to avoid data leakage
    tree_df.loc[:, tree_df.filter(like='from_').columns] = tree_df.filter(like='from_').fillna(0)
    tree_df.loc[:, tree_df.filter(like='lang_').columns] = tree_df.filter(like='lang_').fillna(0)
    return tree_df