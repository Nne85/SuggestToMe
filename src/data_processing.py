import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data():
    # Load the u.data file into a pandas DataFrame
    data_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', names=data_columns)
    
    # Convert timestamp to datetime format
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')

    # Define column names for u.item
    item_columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
                    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                    'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    # Load the u.item file into a pandas DataFrame
    movies = pd.read_csv('data/ml-100k/u.item', sep='|', names=item_columns, encoding='latin-1')

    # Remove duplicate movie entries (based on movie_id)
    movies = movies.drop_duplicates(subset='movie_id', keep='first')

    # Extract genre columns
    genre_columns = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                     'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                     'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    # Add a column with a list of genres
    movies['genres'] = movies[genre_columns].apply(lambda row: [genre for genre in genre_columns if row[genre] == 1], axis=1)
    
    # Drop original genre columns
    movies = movies.drop(columns=genre_columns + ['unknown', 'video_release_date', 'IMDb_URL'])

    # Load users data
    user_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    users = pd.read_csv('data/ml-100k/u.user', sep='|', names=user_columns)
    
    # Clean users data
    users['occupation'] = users['occupation'].str.title()

    # Merge ratings with movies on movie_id
    ratings_with_movies = pd.merge(ratings, movies[['movie_id', 'movie_title', 'release_date', 'genres']], on='movie_id')

    # Merge the above result with users on user_id
    full_data = pd.merge(ratings_with_movies, users, on='user_id')

    return ratings_with_movies, full_data

def preprocess_data(ratings_with_movies, full_data):
    # Check for and drop missing values
    print(f"Missing values before drop: {full_data.isnull().sum()}")
    full_data.dropna(inplace=True)
    print(f"Missing values after drop: {full_data.isnull().sum()}")

    print(f"Missing values before drop: {ratings_with_movies.isnull().sum()}")
    ratings_with_movies.dropna(inplace=True)


    # Convert categorical data into numerical data where applicable
    le = LabelEncoder()
    full_data['occupation'] = le.fit_transform(full_data['occupation'])
    full_data['gender'] = le.fit_transform(full_data['gender'])
    
    # Split data into training and testing sets
    train_df, test_df = train_test_split(full_data, test_size=0.2, random_state=42)
    
    return train_df, test_df

def process_data():
    ratings_df, full_data_df = load_data()
    train_df, test_df = preprocess_data(ratings_df, full_data_df)
    
    # Create the directory if it doesn't exist
    output_dir = 'data/processed'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write the dataframes to CSV files
    output_file = os.path.join(output_dir, 'ratings_data.csv')
    train_file = os.path.join(output_dir, 'train_data.csv')
    test_file = os.path.join(output_dir, 'test_data.csv')
    
    try:
        ratings_df.to_csv(output_file, index=False)
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        print(f'Data saved to {output_file}')
        print(f'Train data saved to {train_file}')
        print(f'Test data saved to {test_file}')

    except Exception as e:
        print(f'Error saving data: {e}')

process_data()