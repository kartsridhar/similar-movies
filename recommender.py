import pandas as pd
import numpy as np

# Get the movies and ratings data frame
df_movies = pd.read_csv('data/movies.csv')
df_ratings = pd.read_csv('data/ratings.csv')

# Merging the frames over movieID
df = pd.merge(df_ratings, df_movies, on="movieId")

# Generating a user/movie rating matrix.
# NaN = user did not rate that particular movie
# All this done by pivot_table
rate_mat = df.pivot_table(index="userId", columns="title", values="rating")

# Extracting the ratings for a particular movie
forrestGumpRatings = rate_mat['Forrest Gump (1994)']

# Finding similar movies ==> Correlations using corrwith()
# compute the pairwise correlation of movie vector of rate_mat
# with every other movie
similar = rate_mat.corrwith(forrestGumpRatings)

# Construct a new DataFrame of movies and their correlation score
corrForrest = pd.DataFrame(similar, columns=['Correlation'])

# Remove all the NaN values
corrForrest.dropna(inplace=True)
