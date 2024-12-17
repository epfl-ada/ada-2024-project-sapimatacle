import os
import json

import pandas as pd
import numpy as np
from textblob import TextBlob
from tqdm import tqdm

tqdm.pandas()

def get_kw_dataframe(json_folder_path):
    # list all json files in the folder
    json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
    # combine all json files into a single dataframe
    dicts = [json.load(open(os.path.join(json_folder_path, f))) for f in json_files]
    character_kws = {k: v for d in dicts for k, v in d.items()}
    # Create an empty list to store the rows for the dataframe
    rows = []

    # Iterate over the data to structure it for the dataframe
    for character_id, characters in character_kws.items():
        for character_name, adjectives in characters.items():
            # Create a row with ID, character name, and a string of comma-separated adjectives
            rows.append([character_id, character_name, ','.join(adjectives)])

    # Create a dataframe
    kw_df = pd.DataFrame(rows, columns=['Wikipedia_movie_ID', 'Character Name', 'Adjectives'], )
    kw_df.Adjectives = kw_df.Adjectives.str.lower()
    print("Running sentiment analysis...")
    kw_df["sentiment_score"] = kw_df.Adjectives.progress_apply(lambda x: get_sentiment_score(x.split(",")))
    kw_df["char_name_lower"] = kw_df["Character Name"].str.lower()
    kw_df["Wikipedia_movie_ID"] = kw_df["Wikipedia_movie_ID"].astype(int)
    return kw_df

def get_sentiment_score(adj_list):
    score_sum = 0
    if len(adj_list) == 1 and adj_list[0] == "":
        mean_score = np.nan
    else:
        for adj in adj_list:
            # Create a TextBlob object for the adjective
            blob = TextBlob(adj)
            # Get the sentiment polarity (-1 to 1)
            polarity = blob.sentiment.polarity
            score_sum += polarity
        mean_score = score_sum / len(adj_list)
    return mean_score