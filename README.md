# To be continued or discontinued: investigating secret formula for successful movie franchise

## Abstract

We often hear that the second movie in a franchise is always worse than the first. However, human memories tend to be influenced by nostalgia, and it remains uncertain if this classic dinner table debate stands on tangible evidence. In this project, we aim to settle this debate by analyzing the list of movies and supplemental data from [the CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/personas/). The preliminary questions to this analysis are, "What makes a good franhcise movie?" and “Are franchise movies more profitable than non-franchise movies?”. In search of answers to these questions, we investigate various metrics such as box office revenue, viewer rating, diversity representation etc., for franchise movies and make a contrast to non-franchise movies.

## Data story

Our blog posts are available in the following link: https://clementloyer.github.io/ada-website.github.io/

## Research Questions

Following the objectives discussed in the abstract, here is the list of concrete questions to tackle with:

1. **Do franchise movies degrade in quality and box office revenue as the sequel continues?**

2. **Is there an underlying pattern of features that makes franchise movies successful? If so, what makes franchise movies different from non-franchise movies?**
    1. Do some movie genres achieve higher box office revenue in movie franchise than the others? Is the trend consistent in non-franchise movies as well?

    2. Does the length and the number of movies impact the success of a franchise and how does it evolves in the franchise?

    3. Do actors of certain ethnicity/gender groups play particular personas more frequently? Are they depicted positively (hero/heroine) or negatively (villain) in the movie?

    4. From which regions of the world do most franchises come from, and what are the dominant collaboration fluxes between countries? Does it differ a lot from non-franchised movies? Are there parts of the world that mostly create single movies instead of movie series? Are there some parts of the world that interact more often when creating sagas of movies? Are there recurrent bonds that can be identified? And finally, do some features regarding countries of origin have a link with movie revenue and reviews ?

## Proposed additional dataset

We complement our movie data by merging data from [the movie database (TMDB)](https://www.themoviedb.org/). This community-based movie database offers free API for non-commercial use, and we used the database to identify franchise movies in the CMU dataset. We also queried additional features for each movie to utilize them in our analysis. The table below shows the summary of newly added features:

| **Feature**             | **Description**                                                   |
|--------------------------|-------------------------------------------------------------------|
| `tmdb_id`               | Unique movie ID for TMDB                                         |
| `collection_name`       | Franchise name                                                  |
| `collection_id`         | Unique ID for the franchise                                     |
| `vote_count`            | Total number of votes the movie received on TMDB                |
| `vote_average`          | Average rating score based on user votes on TMDB                |
| `genres`                | Genre(s) the movie belongs to                                   |
| `budget`                | Production budget of the movie                                  |
| `revenue`               | Revenue generated by the movie                                  |
| `run_time`              | Total runtime of the movie in minutes                           |
| `tmdb_origin_country`   | Country where the movie was produced                            |
| `tmdb_original_language`| Primary language of the movie's production                      |

We decided to add this dataset and the mentionned features as they seem relevant to answer our research questions, and that the given data set had some issues, such as the genres proposed, which were very specific (this is shown in results.ipynb), not usable, and that needed to be grouped together.

## Methods

### Q1:

Franchise movies have higher box office revenue than non franchise ones (statistical significance), but the analysis on the box office shows that a positive linear correlation exists between the movie budget and its box office revenue. We did see that multiple data are missing, so using the added data set would be recommended. After putting some though on how to compare the box-office revenue between movies, we realized that as a movie budget and revenue are positively correlated, a comparaison of the ratio of those values towards movies in the franchise would be more accurate. The next steps would be to look at an eventual correlation between this ratio and the reviews ; as well as looking at how these values differ depending on the order of the movies in a franchise, and whether they are statistically significant or not. (102)

### Q2:

To know which underlying pattern of features makes franchise movies successful, we first want to look at the parameters that are at play: we want to see if they are usable for our analysis, and if they have any influence on reviews or revenues at all. After this initial stage of data exploration, we train a decision tree ([`HistGradientBoostingClassifier`](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)) to predict whether the subsequent movie exists for a given movie based on features such as genre, language and gender ratio of actors. Shapley values are calculated to identify features that contribute the best to the prediction and therefore, illustrating patterns common to franchise movies.

<!-- Then, once this is done, we could use decision trees to try to predict whether, depending on the budget, the genre, and all the other columns in our dataset for a given movie, the studio how to produce the more successful second movie choosing the best feature.   -->

### Q2-1

We can look at violin plots of revenue normalized by budget (Q1) depending on the genres, and do the same for non-franchise movies. To know better how the genres interacts, and how their interactions affect the revenue, we could plot a heatmap of how often genres are paired together. We can then see if the ones more frequent in franchise movies perform better or not: if an under-represented genre in the franchise movies subset performs better than others, are they less frequent because they are paired? This can be answered by looking at the plots mentionned above.

We could also do the same for movie reviews to answer these questions: which genres are more appreciated? Are they the same for franchise and non-franchise movies? Is there a link between genre movie production and review?

### Q2-2

To identify patterns in the size and duration of a franchise, a timeline plot was created for all franchises. Due to the large number of franchises, sorting, filtering, and coloring options will be provided. The goal is to subcategorise franchises for more detailed analysis, as examining all franchises together tends to yield less clear results. These results can be observed with the average vote as a function of the franchise length or the average vote within a franchise.
The next goal is to perform different correlation tests in subgroups between parameters of a franchise and their success to determine them.

### Q2-3

By cross-referencing `Actor_ethnicity_Freebase_ID` with Wikidata, we restored >400 unique ethnicity categories. Inspired by racial groups used in [the British](https://www.ethnicity-facts-figures.service.gov.uk/style-guide/ethnic-groups/) and [the USA census](https://www.census.gov/topics/population/race/about.html), we came up with the following 7 racial groups into which we manually map these ethnicity categories:

```
Hispanic, White, Black, Asian, Native American, Middle Eastern, Others
```

With these racial groups, we first looked into the representation of each group in franchise and non-franchise movies. Second, adjectives describing movie characters were extracted from corresponding movie plots, and mean sentiment scores of these adjectives were assigned to each character using [TextBlob](https://textblob.readthedocs.io/en/dev/). For adjective extraction, we relied on [GPT-4o mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) by providing the following prompts via [Open AI API](https://github.com/openai/openai-python). The resulting json files are saved in `data/character_kws`.

**System prompt (affects all responses from the model)**:
> Given a list of character names and a movie plot summary, return a JSON object where each character name is a key, and the value is a list of adjectives that describe the character. Do not repeat the same words in the list. If a character is not mentioned or described in the plot, return an empty list for that character. The output should be directly loaded by json.loads() in Python.

**Individual prompt (given for each movie plot)**
> Character names: {characters} \nMovie plot summary: {plot}

### Q2-4

The idea is to compare two networked maps of the world; one considering only franchised movies, the other considering all movies in the dataset. 

The maps show one node for each country of the dataset (or regions for more clarity) and the connections between them. For each movie with a pair of origin countries, a connection is created. When a movie has multiple countries of origin, multiple pairs are created.

Do connections increase the box office revenue of the movies? Is the effect significant? And significantly different from non-franchise movies?

## Proposed timeline

### Until P2 (Nov. 15):

The preliminary analysis of the different data we have such as genre, box office revenue, character data, etc.

### Nov. 15 - Nov. 22:

We separate work as described in the next section, and continue working on the analyis to answer every single aspect of our research questions.

### Nov. 22-29:

Start creating the decision tree, reflect on other possible features to create the best second movie, or sequel, that might be valuable.

### Nov. 29 - Dec. 6:

A week dedicated to HW2. No work on the project.

### Dec. 6-13:

Some of us will start creating the website, the plan is to use Jekyll for the desing. All the visualization we use in the website should be finalized by the end of this week.

### Dec. 13-20:

Writing the data story, remarks and conclusion of the project on the web site

## **Organization within the team**

- **Takuya** was in charge of downloading TMDB dataset and also preprocessing and the preliminary analysis of character data
- **Maylis** will work on the movie genres and their subsequent analysis
- **Salomé** will work on analyzing if there is a correlation between box office revenue, budget, its ratio and other feature of a movie.
- **Clément** will be in charge of finalising the research about countries and regions, and will directly focus on building the website using Jekyll.
- **Pierre** will work on the timeline visualisation and the research of the significant parameter and their influence.

## Usage

### Package installation
From the project root, please run:
```
conda install --file requirements.txt
```

### Fetching TMDB data

1. Make sure that `movie.metadata.tsv` is in `data/`. The CMU dataset can be downloaded from [this link](https://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz).

2. Obtain an API key from TMDB. Please follow the instruction on [this webpage](https://developer.themoviedb.org/docs/getting-started).

3. Create `data/constants.py` and add the following:

```python

API_KEY = "YOUR_API_KEY"

```

4. From the root, run `python fetch_data_from_tmdb.py`. This will create `data/movie_metadata_with_tmdb.csv`. Note that the run will take 2-3 hours, depending on your Internet connection.