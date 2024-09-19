# app_ui/streamlit_app.py
import streamlit as st
import pandas as pd
from surprise import Reader, Dataset, KNNWithMeans
from surprise.model_selection import train_test_split

# Load ratings data
@st.cache_data
def load_cached_data():
    return pd.read_csv('data/processed/ratings_data.csv')

ratings_df = load_cached_data()

# Define ratings dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)

# Split data into training and testing sets
trainset, testset = train_test_split(data, test_size=.2)

# Train KNNWithMeans model
@st.cache_resource
def train_model():
    sim_matrix = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': False})
    return sim_matrix.fit(trainset)

sim_matrix = train_model()

# Function to get movie genres
def get_movie_genres(movie_id):
    return ratings_df.loc[ratings_df['movie_id'] == movie_id, 'genres'].iloc[0]

# Function to calculate genre similarity
def genre_similarity(movie1, movie2):
    genres1 = set(movie1.split('|'))
    genres2 = set(movie2.split('|'))
    return len(genres1 & genres2) / len(genres1 | genres2)

# Function to get recommendations
def get_recommendations(user_id, liked_movie_id, num_recs=10):
    try:
        # Get liked movie genres
        liked_movie_genres = get_movie_genres(liked_movie_id)
        
        # Get user's rated movies
        user_rated_movies = ratings_df.loc[ratings_df['user_id'] == user_id, 'movie_id'].tolist()
        
        # Calculate similarities using KNNWithMeans
        similarities = sim_matrix.get_neighbors(liked_movie_id, k=num_recs*2)
        
        # Calculate genre-based similarities
        genre_recs = {}
        for movie_id in similarities:
            if movie_id not in user_rated_movies:
                movie_genres = get_movie_genres(movie_id)
                genre_recs[movie_id] = genre_similarity(liked_movie_genres, movie_genres)
        
        # Combine similarities
        recs = sorted(genre_recs.items(), key=lambda x: x[1], reverse=True)[:num_recs]
        
        # Return recommended movie IDs
        return [rec[0] for rec in recs]
    
    except Exception as e:
        return f"Error: {e}"

def main():
    st.title("Movie Recommendation System")

    user_id = st.number_input("Enter your user ID", min_value=1, max_value=943, value=1, key="user_id_input")

    # Create a list of unique movie titles
    movie_titles = ratings_df['movie_title'].unique().tolist()
    movie_titles.sort()

    liked_movie_title = st.selectbox("Select a movie you liked", options=movie_titles, key="movie_select")

    # Map liked movie title to movie ID
    liked_movie_id = ratings_df.loc[ratings_df['movie_title'] == liked_movie_title, 'movie_id'].iloc[0]

    if st.button("Get Recommendations", key="recommend_button"):
        with st.spinner("Generating recommendations..."):
            recs = get_recommendations(user_id, liked_movie_id)
        
        if isinstance(recs, str):
            st.error(recs)
        else:
            st.write("Recommended movies:")
            for i, rec in enumerate(recs):
                movie_title = ratings_df.loc[ratings_df['movie_id'] == rec, 'movie_title'].iloc[0]
                st.write(f"{i+1}. {movie_title}")

if __name__ == "__main__":
    main()