# To be continued or discontinued: investigating secret formula for successful movie franchise

## Abstract

We often hear that the second movie in a franchise is always worse than the first. However, human memories tend to be influenced by nostalgia, and it remains uncertain if this classic dinner table debate stands on tangible evidence. In this project, we aim to settle this debate by analyzing the list of movies and supplemental data from [the CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/personas/). The preliminary question to this analysis is, "What makes a good movie?" Are franchise movies more profitable than non-franchise movies? In search of answers to these questions, we investigate various metrics such as box office revenue, viewer rating, diversity representation etc., for franchise movies and make a contrast to non-franchise movies.

## Research Questions

Following the objectives discussed in the abstract, here is the list of concrete questions to tackle with:

Q1: Do franchise movies degrade in quality and box office revenue as the sequel continues?

Q2: Is there an underlying pattern of features that makes franchise movies successful?

1. Do some movie genres achieve higher box office revenue in moive franchise than the others? Is the trend consistent in non-franchise movies as well?

2. What are the features that are most useful in predicting box office revenue?

3. Do actors of certain ethnicity groups play particular personas more frequently? Are they depicted positively (hero/heroine) or negatively (villain) in the movie?
  
## Proposed additional dataset

We complement our movie data by merging data from [the movie database (TMDB)](https://www.themoviedb.org/). This community-based movie database offers free API for non-commercial use, and we used the database to identify franchise movies in the CMU dataset. We also queried additional features for each movie to utilize them in our analysis. The table below shows the summary of newly added features:

| **Feature**                | **Description**                                                   |
|--------------------------|---------------------------------------------------------------|
| `tmdb_id`                | Unique movie ID for TMDB     |
| `collection_name`        | Franchise name|
| `collection_id`          | Unique ID for the franchise|
| `vote_count`             | Total number of votes the movie received on TMDB               |
| `vote_average`           | Average rating score based on user votes on TMDB                |
| `genres`                 | Genre(s) the movie belongs to                             |
| `budget`                 | Production budget of the movie                            |
| `revenue`                | Revenue generated by the movie                            |
| `run_time`               | Total runtime of the movie in minutes                     |
| `tmdb_origin_country`    | Country where the movie was produced                      |
| `tmdb_original_language` | Primary language of the movie's production                |

## Methods

## Proposed timeline

## Organization within the team
* Takuya will work on

## Questions for TAs (optional)

## Usage

### Package installation

Conda installation

### Fetching TMDB data

1. Make sure that `movie.metadata.tsv` is in `data/`. The CMU dataset can be downloaded from [this link](https://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz).
2. Obtain an API key from TMDB. Please follow the instruction on [this webpage](https://developer.themoviedb.org/docs/getting-started).
3. Create `data/constants.py` and add the following:

```python
API_KEY = "YOUR_API_KEY"
```

4. From the root, run `python fetch_data_from_tmdb.py`. This will create `data/movie_metadata_with_tmdb.csv`. Note that the run will take 2-3 hours, depending on your Internet connection.
