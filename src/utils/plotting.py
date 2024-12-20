from collections import Counter
from itertools import combinations

import warnings
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from ..data.utils import create_is_from_asia, custom_autopct

# Suppress FutureWarnings and UserWarnings
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=UserWarning)

def plot_network(franchise_df):
    ### 1 - Cleaning the data   
    # Initialize an empty list to store countries
    cleaned_countries = []
    country_counts = Counter()

    # Iterate
    for movie in franchise_df['Movie countries (Freebase ID:name tuples)']:
        if pd.notnull(movie):
            # Split the country by comma and strip any extra characters
            country_list = [movie.split(":")[1].strip().replace("\"", "").replace("}", "") for movie in movie.split(",") if ":" in movie]
            # Replace 'Hong Kong' with 'China'
            country_list = ['China' if country == 'Hong Kong' else country for country in country_list]
            country_list = ['Germany' if country == 'West Germany' else country for country in country_list]
            country_list = ['Germany' if country == 'Weimar Republic' else country for country in country_list]
            country_list = ['Germany' if country == 'German Democratic Republic' else country for country in country_list]
            country_list = ['Malaysia' if country == 'Singapore' else country for country in country_list]
            country_list = ['United Kingdom' if country == 'Kingdom of Great Britain' else country for country in country_list]
            country_list = ['United Kingdom' if country == 'England' else country for country in country_list]
            country_list = ['Croatia' if country == 'Yugoslavia' else country for country in country_list]
            country_list = ['Czechia' if country == 'Czech Republic' else country for country in country_list]
            country_list = ['Czechia' if country == 'Czechoslovakia' else country for country in country_list]
            country_list = ['Czechia' if country == 'Federal Republic of Yugoslavia' else country for country in country_list]
            country_list = ['Russia' if country == 'Soviet Union' else country for country in country_list]
            country_list = ['United Kingdom' if country == 'Scotland' else country for country in country_list]
            country_list = ['United Kingdom' if country == 'Wales' else country for country in country_list]
            country_list = ['United Kingdom' if country == 'Northern Ireland' else country for country in country_list]
            country_list = ['North Macedonia' if country == 'Republic of Macedonia' else country for country in country_list]
            country_list = ['Germany' if country == 'Nazi Germany' or country == 'German Language' else country for country in country_list]
            country_list = ['Slovakia' if country == 'Slovak Republic' else country for country in country_list]
            country_list = ['Serbia' if country == 'Bosnia and Herzegovina' else country for country in country_list]
            country_list = ['Italy' if country == 'Kingdom of Italy' else country for country in country_list]
            country_list = ['South Korea' if country == 'Korea' else country for country in country_list]
            country_list = ['France' if country == 'Monaco' else country for country in country_list]
            country_list = ['Italy' if country == 'Malta' else country for country in country_list]
            country_list = ['Croatia' if country == 'Socialist Federal Republic of Yugoslavia' else country for country in country_list]
            country_list = ['Serbia' if country == 'Serbia and Montenegro' else country for country in country_list]
            country_list = ['United Kingdom' if country == 'Isle of Man' else country for country in country_list]
            country_list = ['Uzbekistan' if country == 'Uzbek SSR' else country for country in country_list]
            country_list = ['Palestine' if country == 'Mandatory Palestine' else country for country in country_list]
            country_list = ['Palestine' if country == 'Palestinian territories' or country == 'Palestinian Territories' else country for country in country_list]
            country_list = ['Myanmar' if country == 'Burma' else country for country in country_list]
            country_list = ['Germany' if country == 'Soviet occupation zone' else country for country in country_list]
            country_list = ['Ukraine' if country == 'Ukrainian SSR' or country == 'Ukranian SSR' else country for country in country_list]
            country_list = ['Georgia' if country == 'Georgian SSR' else country for country in country_list]
            #country_list = ['Kazakhstan' if country == 'Kazakh SSR' else country for country in country_list]
            #country_list = ['Kyrgyzstan' if country == 'Kirghiz SSR' else country for country in country_list]
            country_list = ['Congo' if country == 'Democratic Republic of the Congo' else country for country in country_list]
            country_list = ['Iraq' if country == 'Iraqi Kurdistan' else country for country in country_list]
            country_list = ['Venezuela' if country == 'Aruba' else country for country in country_list]
            country_list = ['China' if country == 'Republic of China' or country == 'Macau' else country for country in country_list]
            country_list = ['India' if country == 'Malayalam Language' else country for country in country_list]

            country_counts.update(country_list)
            cleaned_countries.append(country_list)

    country_counts_df = pd.DataFrame.from_dict(country_counts, orient='index', columns=['counts']).reset_index()
    country_counts_df.columns = ['country', 'counts']

    ### Create pairs of countries
    country_pairs =[]

    for countries in cleaned_countries:
        if len(countries) > 1:
            # Generate all possible combinations of countries
            country_pairs.extend(list(combinations(countries, 2)))

    ### Prepare the world map
    # Load world map data
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Reproject the world map to a suitable projected CRS. The original data are for spherical coordinates and not planar.
    #world_projected = world.to_crs(epsg=3857)  # Using Web Mercator projection (EPSG:3857)

    # Make a list keeping only each country once
    unique_countries = set(country for countries in cleaned_countries for country in countries)
    unique_countries = list(unique_countries)


    #Get the centroid of each country
    country_coordinates={}

    for country in unique_countries:
        country_data = world[world['name'] == country]
        if not country_data.empty:
            # Get the centroid of the country
            centroid = country_data.geometry.centroid.iloc[0]
            country_coordinates[country] = (centroid.x, centroid.y)


    # Plot the graph 
    # Initialize the graph
    G = nx.Graph()

    # Add nodes with geographic positions
    for country, (x, y) in country_coordinates.items():
        G.add_node(country, pos=(x, y))

    # Add edges
    for pair in country_pairs:
        G.add_edge(pair[0], pair[1])

    # Extract positions for networkx
    pos = nx.get_node_attributes(G, 'pos')

    # Plot the world map
    fig, ax = plt.subplots(figsize=(20, 10))
    world.plot(ax=ax, color="lightgrey")

    # Draw the network on top of the world map

    # Get the size of each node based on the country count
    node_sizes = [country_counts[country] * 1 for country in G.nodes()]

    nx.draw(
        G, pos, ax=ax, with_labels=False, node_color='lightblue',
        node_size=node_sizes, font_size=8, font_weight='bold', edge_color='black'
    )
    # Set the title
    plt.title("Geographical Network of Film Industry Connections", fontsize=14)
    plt.show()

def plot_geo_piecharts(movie_df, franchise_df, movies_no_franchise_df):
    # Process the list of countries for franchise movies
    list_of_countries = franchise_df.tmdb_origin_country.fillna('')
    for i in range(len(list_of_countries)):
        list_of_countries.iloc[i] = list_of_countries.iloc[i].strip("[]").replace("'", "").split(", ")

    # Flatten the list of countries and count occurrences
    country_counts = list_of_countries.explode().value_counts().drop("")

    # Group the counts of countries outside the top 7 into 'Others'
    top_countries = country_counts.nlargest(7)
    other_countries_count = country_counts.iloc[7:].sum()
    top_countries['Others'] = other_countries_count

    # Process the list of countries for non-franchise movies
    list_of_countries_non_fr = movies_no_franchise_df.tmdb_origin_country.fillna('')
    for i in range(len(list_of_countries_non_fr)):
        list_of_countries_non_fr.iloc[i] = list_of_countries_non_fr.iloc[i].strip("[]").replace("'", "").split(", ")

    # Flatten the list of countries and count occurrences
    country_counts_non_fr = list_of_countries_non_fr.explode().value_counts().drop("")

    # Group the counts of countries outside the top 7 into 'Others'
    top_countries_non_fr = country_counts_non_fr.nlargest(7)
    other_countries_count_non_fr = country_counts_non_fr.iloc[7:].sum()
    top_countries_non_fr['Others'] = other_countries_count_non_fr

    # Plot the pie charts side by side
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    # Plot for franchise movies
    top_countries.plot(kind='pie', ax=axes[0], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("colorblind", len(top_countries)))
    axes[0].set_ylabel('')
    axes[0].set_title('Top 7 Countries in Franchise', fontsize=14)

    # Plot for non-franchise movies
    top_countries_non_fr.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("colorblind", len(top_countries_non_fr)))
    axes[1].set_ylabel('')
    axes[1].set_title('Top 7 Countries in Non-Franchise', fontsize=14)

    fr_df = create_is_from_asia(franchise_df)
    non_fr_df = create_is_from_asia(movies_no_franchise_df)
    # Calculate proportions for both franchise and non-franchise movies
    asia_prop_fr = fr_df['is_from_asia'].value_counts()
    asia_prop_non_fr = non_fr_df['is_from_asia'].value_counts()

    # Plot franchise movies
    asia_prop_fr.plot(kind='pie', ax=axes[2], autopct=custom_autopct(asia_prop_fr), 
                    labels=['Non-Asian', 'Asian'], colors=['#1f77b4', '#ff7f0e'], startangle=90)
    axes[2].set_title('Franchise Movies from Asia', fontsize=14)
    axes[2].set_ylabel('')

    # Plot non-franchise movies  
    asia_prop_non_fr.plot(kind='pie', ax=axes[3], autopct=custom_autopct(asia_prop_non_fr),
                        labels=['Non-Asian', 'Asian'], colors=['#1f77b4', '#ff7f0e'], startangle=90)
    axes[3].set_title('Non-Franchise Movies from Asia', fontsize=14)
    axes[3].set_ylabel('')

    plt.suptitle('Movies by geography', y=1.05, fontsize=18)
    plt.tight_layout()
    plt.show()

def plot_heatmap_1(KNN_data_filt):
    # Summing country mentions by cluster
    country_sums = KNN_data_filt.groupby('Cluster')[
        [col for col in KNN_data_filt.columns if col.startswith('country_')]
    ].sum()

    # Remove the 'country_' prefix from the column names
    country_sums.columns = country_sums.columns.str.replace('country_', '')

    # Sort countries by total mentions (sum across clusters)
    country_sums = country_sums.loc[:, country_sums.sum().sort_values(ascending=False).index]
    # remove the country that are mentionned only once
    country_sums = country_sums.loc[:, country_sums.sum() > 1]


    # Summing genre mentions by cluster
    genre_sums = KNN_data_filt.groupby('Cluster')[
        [col for col in KNN_data_filt.columns if col.startswith('genre_')]
    ].sum()

    # Remove the 'genre_' prefix from the column names
    genre_sums.columns = genre_sums.columns.str.replace('genre_', '')

    # put a capital letter at the beginning of the genre
    genre_sums.columns = genre_sums.columns.str.capitalize()

    # manually sorting the row to make the heatmap more readable
    genre_sums = genre_sums.loc[:, ['Comedy', 'Family', 'Adventure', 'Horror', 'Thriller', 'Drama', 'Action', 'Animation', 'Fantasy', 'Romance', 'Crime', 'Mystery', 'Music', 'History', 'War', 'Western']]



    # Plot the heatmap
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    sns.heatmap(
        genre_sums.T, 
        annot=True, 
        cmap='YlGnBu', 
        fmt='g', 
        cbar_kws={'label': 'Number of Mentions'},
        ax=axs[0]
    )
    axs[0].set_title('Genre Mentions by Cluster (Sorted by Total Mentions)')
    axs[0].set_xlabel('Cluster')
    axs[0].set_ylabel('Genre')

    sns.heatmap(
        country_sums.T, 
        annot=True, 
        cmap='YlGnBu', 
        fmt='g', 
        cbar_kws={'label': 'Number of Mentions'},
        ax=axs[1]
    )
    axs[1].set_title('Country Mentions by Cluster (Sorted by Total Mentions)')
    axs[1].set_xlabel('Cluster')
    axs[1].set_ylabel('Country')
    plt.tight_layout()
    plt.show()

def plot_heatmap_2(KNN_1_2_filt):
    """Function to plot heatmaps"""
    # Summing country of the 1st movie mentions by cluster
    country_sums_1 = KNN_1_2_filt.groupby('Cluster')[
        [col for col in KNN_1_2_filt.columns if col.startswith('tmdb_origin_country_1_')]
    ].sum()

    # Remove the 'country_' prefix from the column names
    country_sums_1.columns = country_sums_1.columns.str.replace('tmdb_origin_country_1_', '')

    # Sort countries by total mentions (sum across clusters)
    country_sums_1 = country_sums_1.loc[:, country_sums_1.sum().sort_values(ascending=False).index]


    # Summing country of the 2nd movie mentions by cluster
    country_sums_2 = KNN_1_2_filt.groupby('Cluster')[
        [col for col in KNN_1_2_filt.columns if col.startswith('tmdb_origin_country_2_')]
    ].sum()

    # Remove the 'country_' prefix from the column names
    country_sums_2.columns = country_sums_2.columns.str.replace('tmdb_origin_country_2_', '')

    # Sort countries by total mentions (sum across clusters)
    country_sums_2 = country_sums_2.loc[:, country_sums_2.sum().sort_values(ascending=False).index]

    # remove the country that are mentionned only once
    country_sums_1 = country_sums_1.loc[:, country_sums_1.sum() > 1]
    country_sums_2 = country_sums_2.loc[:, country_sums_2.sum() > 1]

    for column in country_sums_1.columns:
        if column not in country_sums_2.columns:
            country_sums_2[column] = 0
    country_sums_2 = country_sums_2[country_sums_1.columns]

    # Interleave the columns (1st movie then 2nd movie for each cluster)
    interleaved_columns = []
    for cluster in country_sums_1.columns:
        interleaved_columns.append(country_sums_1[cluster].rename(f"{cluster} 1st Movie"))
        interleaved_columns.append(country_sums_2[cluster].rename(f"{cluster} 2nd Movie"))

    # Combine into a single DataFrame
    country_sums = pd.concat(interleaved_columns, axis=1)


    # Summing genre mentions by cluster
    genre_sums_1 = KNN_1_2_filt.groupby('Cluster')[
        [col for col in KNN_1_2_filt.columns if col.startswith('genre_1_')]
    ].sum()

    # Remove the 'genre_' prefix from the column names
    genre_sums_1.columns = genre_sums_1.columns.str.replace('genre_1_', '')

    # Sort genres by total mentions (sum across clusters)
    genre_sums_1 = genre_sums_1.loc[:, genre_sums_1.sum().sort_values(ascending=False).index]

    # Summing genre mentions by cluster
    genre_sums_2 = KNN_1_2_filt.groupby('Cluster')[
        [col for col in KNN_1_2_filt.columns if col.startswith('genre_2_')]
    ].sum()

    # Remove the 'genre_' prefix from the column names
    genre_sums_2.columns = genre_sums_2.columns.str.replace('genre_2_', '')

    # Sort genres by total mentions (sum across clusters)
    genre_sums_2 = genre_sums_2.loc[:, genre_sums_2.sum().sort_values(ascending=False).index]

    # Interleave the columns (1st movie then 2nd movie for each cluster)
    interleaved_columns = []
    for cluster in genre_sums_1.columns:
        interleaved_columns.append(genre_sums_1[cluster].rename(f"{cluster} 1st Movie"))
        interleaved_columns.append(genre_sums_2[cluster].rename(f"{cluster} 2nd Movie"))

    # Combine into a single DataFrame
    genre_sums = pd.concat(interleaved_columns, axis=1)

    # separate the negative and positive cluster
    genre_sums_neg = genre_sums.T.iloc[:,0:3]
    genre_sums_pos = genre_sums.T.iloc[:,4:10]


    # Plot the heatmap
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    sns.heatmap(
        genre_sums_neg, 
        annot=True, 
        cmap='YlGnBu', 
        fmt='g', 
        cbar_kws={'label': 'Number of Mentions'},
        ax=axs[0]
    )
    axs[0].set_title('Genre Mentions by Cluster (Sorted by Total Mentions)')
    axs[0].set_xlabel('Cluster')
    axs[0].set_ylabel('Genre')

    sns.heatmap(
        genre_sums_pos, 
        annot=True, 
        cmap='YlGnBu', 
        fmt='g', 
        cbar_kws={'label': 'Number of Mentions'},
        ax=axs[1]
    )
    axs[0].set_title('Genre Mentions by Cluster (Sorted by Total Mentions)')
    axs[0].set_xlabel('Cluster')
    axs[0].set_ylabel('Genre')
