import os
import zipfile
import urllib.request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration & Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = os.path.join(DATA_DIR, "ml-100k.zip")
EXTRACT_PATH = os.path.join(DATA_DIR, "ml-100k")

def download_and_extract_data():
    """Downloads and extracts the MovieLens 100k dataset if not present."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # Check if files already exist to avoid re-downloading
    movies_file = os.path.join(DATA_DIR, "movies.csv")
    if os.path.exists(movies_file):
        return

    print("Data files not found. Downloading MovieLens 100k dataset...")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
        
    # Clean up zip file
    os.remove(ZIP_PATH)
    
    # Process the raw ml-100k files into clean CSVs
    process_raw_movielens_files()

def process_raw_movielens_files():
    """Converts raw MovieLens pipeline-separated files into clean CSV format."""
    print("Processing raw data files into clean CSVs...")
    
    # Process Item/Movie file
    item_path = os.path.join(EXTRACT_PATH, "u.item")
    movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                  'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                  'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    # Using latin-1 encoding as u.item contains special characters
    movies_df = pd.read_csv(item_path, sep='|', names=movie_cols, encoding='latin-1')
    
    # Create a string representation of genres for each movie
    genres_list = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                   'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                   'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    def extract_genres(row):
        active_genres = [genre for genre in genres_list if row[genre] == 1]
        return " ".join(active_genres)
    
    movies_df['genres'] = movies_df.apply(extract_genres, axis=1)
    
    # Drop unnecessary structural genre columns and URLs
    keep_cols = ['movie_id', 'title', 'genres']
    movies_df = movies_df[keep_cols]
    movies_df.to_csv(os.path.join(DATA_DIR, "movies.csv"), index=False)
    
    # Process Ratings file
    data_path = os.path.join(EXTRACT_PATH, "u.data")
    rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(data_path, sep='\t', names=rating_cols)
    ratings_df.to_csv(os.path.join(DATA_DIR, "ratings.csv"), index=False)
    print("CSVs generated successfully inside data/ folder.")

# --- Content-Based Engine Class ---
class ContentBasedRecommender:
    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None

    def fit(self):
        """Prepares data, builds TF-IDF matrix and calculates similarity profiles."""
        download_and_extract_data()
        
        # Load the newly processed clean dataset
        self.movies_df = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
        
        # In this basic dataset, metadata is composed solely of 'genres'.
        # For advanced setups, you can append overview text, keywords, etc. here.
        self.movies_df['metadata_soup'] = self.movies_df['genres'].fillna('')

        # 1. Initialize TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        
        # 2. Construct the TF-IDF matrix
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['metadata_soup'])
        
        # 3. Compute the Cosine Similarity matrix
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # 4. Construct a reverse map of indices and movie titles
        # Standardizing titles to lowercase avoids case-sensitivity matching issues
        self.movies_df['title_lower'] = self.movies_df['title'].str.lower().str.strip()
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['title_lower']).to_dict()
        print("Content-based filtering system trained successfully.")

    def get_recommendations(self, title, top_n=10):
        """Fetches the top N most similar movies based on metadata similarity."""
        clean_title = str(title).lower().strip()
        
        if clean_title not in self.indices:
            return f"Movie '{title}' not found in the database database."
            
        # Get the internal index of the target movie
        idx = self.indices[clean_title]
        
        # Pairwise similarity scores of all movies with our target movie
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort the movies based on similarity scores in descending order
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Extract metadata scores of the top items (ignoring the entry item itself)
        sim_scores = sim_scores[1:top_n + 1]
        
        # Map item indices back to their names
        movie_indices = [i[0] for i in sim_scores]
        
        return self.movies_df[['title', 'genres']].iloc[movie_indices].reset_index(drop=True)

# --- Quick Engine Verification ---
if __name__ == "__main__":
    recommender = ContentBasedRecommender()
    recommender.fit()
    
    # Test case engine verification using a known entity pattern
    test_movie = "Toy Story (1995)"
    print(f"\nRecommendations for '{test_movie}':")
    print(recommender.get_recommendations(test_movie, top_n=5))