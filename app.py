import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

# Set page configuration
st.set_page_config(
    page_title="AI Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .recommendation-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">🎬 AI Movie Recommender</h1>', unsafe_allow_html=True)
st.markdown("### Your Personal Cinema Guide")
st.write("""
This intelligent system uses sophisticated machine learning techniques to help you discover movies 
you'll love, eliminating decision fatigue and wasted scrolling time.
""")

# Load data (you'll need to replace these with your actual datasets)
@st.cache_data
def load_data():
    # Sample data - replace with your actual movies.csv and ratings.csv
    movies_data = {
        'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'title': ['Galactic Wars', 'The Space Between', 'Fast Lane', 'Mystery Island', 
                 'The Last Heist', 'Haunted Manor', 'Ocean Deep', 'Mountain High', 
                 'City Lights', 'Desert Dreams'],
        'genres': ['Action|Sci-Fi', 'Drama|Romance', 'Action|Thriller', 'Adventure|Mystery',
                  'Action|Crime', 'Horror|Thriller', 'Adventure|Drama', 'Adventure|Drama',
                  'Drama|Romance', 'Adventure|Drama'],
        'description': [
            'Epic space battle for the fate of the galaxy',
            'A story of love across dimensions',
            'High-speed car chases and heists',
            'Ancient mysteries on a remote island',
            'The final job of a retired thief',
            'Ghosts haunt an old mansion',
            'Deep sea exploration adventure',
            'Mountain climbing expedition',
            'Urban love story in a metropolis',
            'Journey through the desert'
        ]
    }
    
    ratings_data = {
        'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'movieId': [1, 3, 5, 2, 4, 6, 1, 7, 9, 3, 8, 10],
        'rating': [5, 4, 3, 5, 4, 2, 4, 5, 3, 4, 4, 5]
    }
    
    movies_df = pd.DataFrame(movies_data)
    ratings_df = pd.DataFrame(ratings_data)
    
    return movies_df, ratings_df

# Load the data
movies_df, ratings_df = load_data()

# Create tabs for different recommendation methods
tab1, tab2, tab3 = st.tabs(["Content-Based Filtering", "Collaborative Filtering", "About the Project"])

with tab1:
    st.markdown('<h2 class="sub-header">🎯 Content-Based Recommendations</h2>', unsafe_allow_html=True)
    st.write("Get recommendations based on movie characteristics like genre and description.")
    
    # Movie selection for content-based filtering
    selected_movie = st.selectbox("Select a movie you like:", movies_df['title'].tolist())
    
    if st.button("Get Content-Based Recommendations"):
        # Create TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        
        # Combine genres and description for better features
        movies_df['content'] = movies_df['genres'] + ' ' + movies_df['description']
        
        # Construct the TF-IDF matrix
        tfidf_matrix = tfidf.fit_transform(movies_df['content'])
        
        # Compute the cosine similarity matrix
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        # Get the index of the movie that matches the title
        idx = movies_df[movies_df['title'] == selected_movie].index[0]
        
        # Get the pairwise similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the scores of the 5 most similar movies (skip the first one as it's the same movie)
        sim_scores = sim_scores[1:6]
        
        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return the top 5 most similar movies
        recommendations = movies_df.iloc[movie_indices][['title', 'genres', 'description']]
        
        st.success(f"Because you liked **{selected_movie}**, you might also enjoy:")
        
        for i, row in recommendations.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="recommendation-box">
                    <h4>{row['title']}</h4>
                    <p><strong>Genres:</strong> {row['genres']}</p>
                    <p><strong>Description:</strong> {row['description']}</p>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown('<h2 class="sub-header">👥 Collaborative Filtering</h2>', unsafe_allow_html=True)
    st.write("Get recommendations based on users with similar tastes.")
    
    # User selection for collaborative
