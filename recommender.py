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

movie_name = input("Enter a movie name: ")

# Extracting the ratings for the respective movie
movieRatings = rate_mat[movie_name]

# Finding similar movies ==> Correlations using corrwith()
# compute the pairwise correlation of movie vector of rate_mat
# with every other movie
similarMovies = rate_mat.corrwith(movieRatings)

# Construct a new DataFrame of movies and their correlation score
df_corr = pd.DataFrame(similarMovies)

# Remove all the NaN values
df_corr.dropna(inplace=True)

# Creating a DataFrame counting the number of ratings and overall mean
df_stats = df.groupby('title').agg({'rating': [np.size, np.mean]})

# Choosing movies only with over 200 ratings
popularMovies = df_stats['rating']['size'] >= 200

# Sorting the mean ratings in descending order, choosing the first 10 movies
df_stats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:10]

# Joining the above data to the similarMovies set
df_result = df_stats[popularMovies].join(pd.DataFrame(similarMovies, columns=['Similarity']))

df_result = df_result.sort_values(['Similarity'], ascending=False)[:10]
print(df_result.head())
