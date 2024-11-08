# Importing necessary libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For visualization
import seaborn as sns  # For advanced data visualization
import warnings  # To suppress warnings
from scipy.sparse import csr_matrix  # For efficient matrix storage and operations
from sklearn.neighbors import NearestNeighbors  # For finding similar movies

# Suppress future warnings for cleaner outputs
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the ratings dataset and display its structure
ratings = pd.read_csv("ratings.csv")
ratings.head()  # Show the first few rows of the ratings data

# Load the movies dataset and display its structure
movies = pd.read_csv("movies_rc.csv")
movies.head()  # Show the first few rows of the movies data

# Display information about the ratings dataset
ratings.info()

# Calculate general statistics about the dataset
n_ratings = len(ratings.rating)  # Total number of ratings
n_movies = len(ratings.movieId.unique())  # Number of unique movies
n_users = len(ratings.userId.unique())  # Number of unique users

# Print calculated statistics
print(f'Number of Ratings: {n_ratings}')
print(f'Number of Unique MovieIDs: {n_movies}')
print(f'Number of Unique UserIDs: {n_users}')
print(f'Average ratings per user: {round(n_ratings / n_users, 2)}')
print(f'Average ratings per movie: {round(n_ratings / n_movies, 2)}')

# Analyze the frequency of ratings given by each user
user_freq = ratings[['userId', 'movieId']].groupby('userId').count().reset_index()
user_freq.columns = ['userId', 'n_ratings']  # Rename column for clarity
user_freq.head()

# Calculate the mean rating for each movie
mean_rating = ratings.groupby('movieId')[['rating']].mean()

# Identify the lowest-rated movie
lowest_rated = mean_rating['rating'].idxmin()  # Get the movie ID with the lowest average rating
movies.loc[movies['movieId'] == lowest_rated]  # Retrieve details of the lowest-rated movie

# Identify the highest-rated movie
highest_rated = mean_rating['rating'].idxmax()  # Get the movie ID with the highest average rating
movies.loc[movies['movieId'] == highest_rated]  # Retrieve details of the highest-rated movie

# Count how many users rated the lowest-rated movie
ratings.movieId[ratings['movieId'] == lowest_rated].count()

# Count how many users rated the highest-rated movie
ratings.movieId[ratings['movieId'] == highest_rated].count()

# Function to create a user-item matrix
def create_matrix(df):
    """
    Converts a dataframe into a sparse matrix for collaborative filtering.
    
    Parameters:
    - df: DataFrame containing userId, movieId, and rating columns
    
    Returns:
    - X: Sparse matrix of shape (movies, users)
    - user_mapper, movie_mapper: Maps user/movie IDs to matrix indices
    - user_inv_mapper, movie_inv_mapper: Maps matrix indices back to IDs
    """
    N = len(df['userId'].unique())  # Number of unique users
    M = len(df['movieId'].unique())  # Number of unique movies

    # Map user and movie IDs to matrix indices
    user_mapper = dict(zip(np.unique(df['userId']), list(range(N))))
    movie_mapper = dict(zip(np.unique(df['movieId']), list(range(M))))

    # Map matrix indices back to IDs
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df['userId'])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df['movieId'])))

    # Create matrix indices
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    # Construct the sparse matrix
    X = csr_matrix((df['rating'], (movie_index, user_index)), shape=(M, N))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

# Generate the user-item matrix and supporting mappings
X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

# Function to find similar movies using KNN
def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
    """
    Finds similar movies to the given movie ID using K-Nearest Neighbors.
    
    Parameters:
    - movie_id: ID of the target movie
    - X: User-item sparse matrix
    - k: Number of neighbors to find
    - metric: Distance metric (default: 'cosine')
    - show_distance: Whether to display distances
    
    Returns:
    - List of similar movie IDs
    """
    neighbour_ids = []
    movie_ind = movie_mapper[movie_id]  # Map movie ID to index
    movie_vec = X[movie_ind].reshape(1, -1)  # Retrieve movie vector and reshape for KNN

    knn = NearestNeighbors(n_neighbors=k + 1, algorithm='brute', metric=metric)
    knn.fit(X)  # Fit the KNN model
    neighbour = knn.kneighbors(movie_vec, return_distance=show_distance)  # Find neighbors

    for i in range(0, k + 1):
        n = neighbour[1].item(i)  # Retrieve neighbor index
        neighbour_ids.append(movie_inv_mapper[n])  # Map index back to movie ID

    neighbour_ids.pop(0)  # Remove the movie itself from the list
    return neighbour_ids

# Create a dictionary to map movie IDs to titles
movie_titles = dict(zip(movies.movieId, movies.title))

# Example: Find similar movies for a given movie ID
movie_id = 3
similar_ids = find_similar_movies(movie_id, X, k=10)
movie_title = movie_titles[movie_id]

print(f'Since you watched {movie_title}')
for i in similar_ids:
    print(movie_titles[i])

# Function to recommend movies for a user
def recommend_movies_for_user(user_id, k=10):
    """
    Recommends movies for a specific user based on their highest-rated movie.
    
    Parameters:
    - user_id: ID of the user
    - k: Number of recommendations
    
    Prints:
    - List of recommended movies
    """
    df1 = ratings[ratings.userId == user_id]  # Filter ratings for the user
    if df1.empty:
        print(f'User with ID {user_id} does not exist.')
        return

    movie_id = df1[df1.rating == max(df1.rating)].movieId.iloc[0]  # Get highest-rated movie
    movie_title = movie_titles.get(movie_id, "Movie not found")

    if movie_title == "Movie not found":
        print(f"Movie with ID {movie_id} not found.")
        return

    similar_ids = find_similar_movies(movie_id, X, k)  # Find similar movies
    print(f"Since you watched {movie_title}, you might also like:")
    for i in similar_ids:
        print(movie_titles.get(i, "Movie not found"))

# Example: Recommend movies for a specific user
user_id = 150
recommend_movies_for_user(user_id, k=10)
