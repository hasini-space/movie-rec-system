import os
import streamlit as st
import pandas as pd

# Import the engines we built
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender

# --- Configuration & Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MOVIES_FILE = os.path.join(DATA_DIR, "movies.csv")
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")

# --- Netflix Styling Configuration ---
st.set_page_config(
    page_title="AI Movie suggestion",
    page_icon="🍿",
    layout="wide"
)

# Inject custom CSS for a dark Netflix vibe
st.markdown("""
    <style>
    .main {
        background-color: #141414;
        color: #FFFFFF;
    }
    h1 {
        color: #E50914 !important; /* Netflix Red */
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: bold;
    }
    h2, h3 {
        color: #FFFFFF !important;
    }
    .stButton>button {
        background-color: #E50914 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #B20710 !important;
    }
    /* Movie Card Styling */
    .movie-card {
        background-color: #181818;
        padding: 15px;
        border-radius: 6px;
        border: 1px solid #282828;
        text-align: center;
        height: 100%;
        transition: transform .2s;
    }
    .movie-card:hover {
        transform: scale(1.05);
        border-color: #E50914;
    }
    .movie-title {
        font-size: 16px;
        font-weight: bold;
        color: #FFFFFF;
        margin-bottom: 8px;
    }
    .movie-genre {
        font-size: 12px;
        color: #AAAAAA;
    }
    .movie-match {
        font-size: 14px;
        color: #46D369; /* Netflix Green Match % */
        font-weight: bold;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Engine Cache ---
@st.cache_resource
def initialize_engines():
    cb_engine = ContentBasedRecommender()
    cb_engine.fit()
    cf_engine = CollaborativeRecommender()
    try:
        cf_engine.load_model()
    except FileNotFoundError:
        cf_engine.fit()
    return cb_engine, cf_engine

with st.spinner("Streaming algorithms from internal engine..."):
    cb_system, cf_system = initialize_engines()

movies_list_df = pd.read_csv(MOVIES_FILE)
all_movie_titles = sorted(movies_list_df['title'].unique())

# --- Top Header Navigation ---
st.title("MOVIE TIME *(suggestion)*")
st.markdown("### Hybrid AI-Driven Personalization Dashboard")
st.write("---")

# User Path Selection via Sidebar Profile switching
st.sidebar.title("👤 User Profiles")
ratings_df = pd.read_csv(RATINGS_FILE)
active_user = st.sidebar.number_input(
    "Switch Profile (User ID):", 
    min_value=int(ratings_df['user_id'].min()), 
    max_value=int(ratings_df['user_id'].max()), 
    value=196
)

# --- Main Layout ---
# Row 1: Content-Based Row ("Because you watched...")
st.header("🍿 Because You Watched")
selected_movie = st.selectbox(
    "Select a movie from your viewing history:", 
    options=all_movie_titles,
    index=0
)

cb_recs = cb_system.get_recommendations(selected_movie, top_n=5)

if not isinstance(cb_recs, str):
    # Display recommendations as a horizontal row of cards
    cols = st.columns(5)
    for i, row in cb_recs.iterrows():
        with cols[i]:
            st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">{row['title']}</div>
                    <div class="movie-genre">{row['genres']}</div>
                    <div class="movie-match">98% Match</div>
                </div>
            """, unsafe_allow_html=True)

st.write("---")

# Row 2: Collaborative Row ("Top Picks for You")
st.header(f"🔥 Top Picks for User Profile {active_user}")

cf_recs = cf_system.get_user_recommendations(user_id=active_user, top_n=5)

if not cf_recs.empty:
    cols = st.columns(5)
    for i, row in cf_recs.iterrows():
        with cols[i]:
            # Convert internal SVD 1-5 rating into an arbitrary "Match %" for a real streaming UI feel
            match_percentage = int((row['predicted_rating'] / 5.0) * 100)
            
            st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">{row['title']}</div>
                    <div class="movie-genre">{row['genres']}</div>
                    <div class="movie-match">{match_percentage}% Match</div>
                </div>
            """, unsafe_allow_html=True)