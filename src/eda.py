import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caching the data for performance
@st.cache_data
def load_data():
    # Load processed data (ensure this path and file exist)
    return pd.read_csv('data/processed/ratings_data.csv')

def main():
    st.title("Movie Ratings Exploratory Analysis")

    # Load the data
    data = load_data()

    # Header: Distribution of Movie Ratings
    st.header("Distribution of Movie Ratings")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['rating'], bins=5, kde=True, ax=ax)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Number of Ratings')
    ax.set_title('Distribution of Movie Ratings')
    st.pyplot(fig)

    # Header: Top 10 Most Rated Movies
    st.header("Top 10 Most Rated Movies")
    # Ensure the 'title' column exists, adjust if necessary
    most_rated = data.groupby('movie_title').size().sort_values(ascending=False).head(10)
    st.bar_chart(most_rated)

    # Header: Average Rating for Top 10 Most Rated Movies
    st.header("Average Rating for Top 10 Most Rated Movies")
    top_10_movies = most_rated.index
    avg_ratings = data[data['movie_title'].isin(top_10_movies)].groupby('movie_title')['rating'].mean().sort_values(ascending=False)
    st.bar_chart(avg_ratings)

    # Header: Rating Distribution for Top 5 Most Rated Movies
    st.header("Rating Distribution for Top 5 Most Rated Movies")
    top_5_movies = most_rated.head().index
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='movie_title', y='rating', data=data[data['movie_title'].isin(top_5_movies)], ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('Rating Distribution for Top 5 Most Rated Movies')
    st.pyplot(fig)

    # Header: Correlation between Number of Ratings and Average Rating
    st.header("Correlation between Number of Ratings and Average Rating")
    movie_stats = data.groupby('movie_title').agg({'rating': ['count', 'mean']})
    movie_stats.columns = ['rating_count', 'rating_mean']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='rating_count', y='rating_mean', data=movie_stats, ax=ax)
    ax.set_xlabel('Number of Ratings')
    ax.set_ylabel('Average Rating')
    ax.set_title('Correlation between Number of Ratings and Average Rating')
    st.pyplot(fig)

if __name__ == "__main__":
    main()