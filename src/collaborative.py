import os
import pickle
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# --- Configuration & Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")
MOVIES_FILE = os.path.join(DATA_DIR, "movies.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "svd_model.pkl")

class CollaborativeRecommender:
    def __init__(self):
        self.model = None
        self.trainset = None
        self.movies_df = None
        
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)

    def fit(self):
        """Loads rating metrics, trains the SVD matrix factorization model, and saves it."""
        if not os.path.exists(RATINGS_FILE):
            raise FileNotFoundError(
                f"Missing baseline configurations. Run 'python src/content_based.py' first to initialize data."
            )

        # 1. Load data via standard pandas utilities
        ratings_df = pd.read_csv(RATINGS_FILE)
        self.movies_df = pd.read_csv(MOVIES_FILE)

        # 2. Convert standard DataFrames into Surprise-compatible structures
        # The scale defaults to MovieLens standard 1-5 rating benchmarks
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)

        # 3. Create a train/test split to evaluate predictive capabilities
        train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)
        
        print("Training Collaborative Filtering model (SVD matrix factorization)...")
        # Initialize SVD with balanced hyperparameters
        self.model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
        self.model.fit(train_data)

        # 4. Evaluate operational accuracy
        predictions = self.model.test(test_data)
        rmse = accuracy.rmse(predictions)
        print(f"Model trained successfully. Root Mean Squared Error (RMSE): {rmse:.4f}")

        # 5. Retrain on the complete dataset to maximize prediction coverage
        print("Finalizing weights over the entire user matrix...")
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)

        # 6. Serialize and cache the model object to disk
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model weights successfully exported to: {MODEL_PATH}")

    def load_model(self):
        """Loads pre-trained model structural assets from disk storage."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Pre-trained structural model files not found. Call fit() first.")
        
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        self.movies_df = pd.read_csv(MOVIES_FILE)
        print("Pre-trained model loaded successfully from disk cache.")

    def predict_rating(self, user_id, movie_id):
        """Predicts the exact numerical rating an isolated user would give to a specific movie."""
        if self.model is None:
            self.load_model()
        
        # Returns a prediction object containing the estimated rating ('est')
        prediction = self.model.predict(uid=int(user_id), iid=int(movie_id))
        return prediction.est

    def get_user_recommendations(self, user_id, top_n=10):
        """Discovers the highest-predicted movies for a user that they haven't rated yet."""
        if self.model is None:
            self.load_model()

        # Find movies already explicitly rated by this specific user
        ratings_df = pd.read_csv(RATINGS_FILE)
        user_rated_movies = set(ratings_df[ratings_df['user_id'] == int(user_id)]['movie_id'])

        # Generate predictions across all unrated assets
        all_movie_ids = self.movies_df['movie_id'].unique()
        predictions = []

        for m_id in all_movie_ids:
            if m_id not in user_rated_movies:
                est_rating = self.predict_rating(user_id, m_id)
                predictions.append((m_id, est_rating))

        # Sort based on descending estimated scores
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:top_n]

        # Extract textual titles and structural categories
        rec_movie_ids = [item[0] for item in top_predictions]
        rec_scores = [item[1] for item in top_predictions]

        result_df = self.movies_df[self.movies_df['movie_id'].isin(rec_movie_ids)].copy()
        
        # Set alignment order based on recommendation scores
        result_df['predicted_rating'] = result_df['movie_id'].map(dict(top_predictions))
        return result_df.sort_values(by='predicted_rating', ascending=False).reset_index(drop=True)

# --- Standalone Engine Verification ---
if __name__ == "__main__":
    recommender = CollaborativeRecommender()
    
    # Train and serialize the system
    recommender.fit()
    
    # Quick target validation test (Using arbitrary historical active user ID 196)
    test_user = 196
    print(f"\nTop recommendations for User {test_user}:")
    print(recommender.get_user_recommendations(user_id=test_user, top_n=5))