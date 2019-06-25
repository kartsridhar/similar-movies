import pandas as pd
import numpy as np

# Get the movies and ratings data frame
df_movies = pd.read_csv('data/movies.csv')
df_ratings = pd.read_csv('data/ratings.csv')

# Merging the frames over movieID
df = pd.merge(df_ratings, df_movies)

# Generating a user/movie rating matrix.
# NaN = user did not rate that particular movie
# All this done by pivot_table
rate_mat = df.pivot_table(index="userId", columns="title", values="rating")

# find the correlation matrix. min period means at least 150
corrMatrix = rate_mat.corr(min_periods=150)

user_0_rat = rate_mat.iloc[0].dropna()

# creating a list similar to the ones I rated
simCandidates = pd.Series()
for i in range(0, len(user_0_rat.index)):
    print("Adding sims for " + user_0_rat.index[i] + "...")

    #Retrieve similar movies that I rated
    sims = corrMatrix[user_0_rat.index[i]].dropna()
    sims = sims.map(lambda x : x * user_0_rat[i])
    simCandidates = simCandidates.append(sims)

print("Sorting...")
simCandidates.sort_values(inplace=True, ascending=False)
print(simCandidates.head())
