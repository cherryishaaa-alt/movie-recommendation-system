import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cinema Theater Styling
st.set_page_config(
    page_title="🎭 CineMagic Theater",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Cinema Theater CSS
st.markdown("""
<style>
    /* Main background - dark theater vibe */
    .main {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: white;
    }
    
    /* Theater header with curtain effect */
    .theater-header {
        background: linear-gradient(90deg, #8B0000 0%, #B22222 25%, #DC143C 50%, #B22222 75%, #8B0000 100%);
        padding: 30px;
        text-align: center;
        border-bottom: 5px solid #FFD700;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(139, 0, 0, 0.5);
    }
    
    /* Movie card - like theater screen */
    .movie-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border: 2px solid #FFD700;
        padding: 20px;
        margin: 15px 0;
        border-radius: 10px;
        color: white;
        box-shadow: 0 8px 25px rgba(0, 0, 139, 0.3);
        transition: transform 0.3s ease;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0, 0, 139, 0.5);
    }
    
    /* Spotlight effect for recommendations */
    .spotlight-header {
        background: linear-gradient(90deg, #8B0000 0%, #DC143C 50%, #8B0000 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 25px 0;
        border: 2px solid #FFD700;
        box-shadow: 0 0 30px rgba(220, 20, 60, 0.3);
    }
    
    /* Marquee text effect */
    .marquee-text {
        background: linear-gradient(90deg, #00008B 0%, #1e3c72 50%, #00008B 100%);
        color: #FFD700;
        padding: 15px;
        text-align: center;
        font-weight: bold;
        border: 2px solid #FFD700;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load movie data with error handling
@st.cache_data
def load_data():
    try:
        movies_df = pd.read_csv('movies.csv')
        return movies_df
    except Exception as e:
        st.error(f"❌ Error loading movies.csv: {str(e)}")
        # Return sample data if file doesn't load
        return pd.DataFrame({
            'movieId': [1, 2, 3],
            'title': ['Sample Movie 1', 'Sample Movie 2', 'Sample Movie 3'],
            'genres': ['Action', 'Drama', 'Comedy'],
            'description': ['Description 1', 'Description 2', 'Description 3']
        })

movies_df = load_data()

# Recommendation function
def get_recommendations(movie_title, movies_df, num_recommendations=6):
    try:
        # Combine genres and description
        movies_df['content'] = movies_df['genres'] + ' ' + movies_df['description']
        
        # Create TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(movies_df['content'])
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Get the index of the movie
        idx = movies_df[movies_df['title'] == movie_title].index[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar movies
        sim_scores = sim_scores[1:num_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        
        return movies_df.iloc[movie_indices]
    except Exception as e:
        st.error(f"Error in recommendations: {str(e)}")
        return movies_df.head(num_recommendations)

# Theater Header
st.markdown("""
<div class="theater-header">
    <h1 style="color: #FFD700; font-size: 4rem; margin: 0; text-shadow: 3px 3px 5px rgba(0,0,0,0.5);">🎭 CINEMAGIC THEATER</h1>
    <p style="color: white; font-size: 1.5rem; margin: 10px 0 0 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">Your Personal Cinema Experience</p>
</div>
""", unsafe_allow_html=True)

# Show movie count
st.markdown(f"""
<div class="marquee-text">
    🎬 NOW SHOWING: {len(movies_df)} BLOCKBUSTER MOVIES • HARRY POTTER • AVENGERS • LORD OF THE RINGS • SPIDER-MAN • STAR WARS 🎬
</div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎟️ GET RECOMMENDATIONS", "📽️ BROWSE MOVIES", "🔍 SEARCH"])

with tab1:
    st.markdown('<div class="spotlight-header"><h2>🎯 PERSONALIZED MOVIE RECOMMENDATIONS</h2></div>', unsafe_allow_html=True)
    
    selected_movie = st.selectbox(
        "Choose a movie you love:",
        movies_df['title'].tolist()
    )
    
    num_recommendations = st.slider("Number of recommendations:", 3, 8, 6)
    
    if st.button("🚀 FIND SIMILAR MOVIES", use_container_width=True):
        with st.spinner('🎭 Searching our theater database...'):
            recommendations = get_recommendations(selected_movie, movies_df, num_recommendations)
        
        st.balloons()
        st.markdown(f'<div class="spotlight-header"><h3>🎉 BECAUSE YOU LIKED: <b style="color:#FFD700">"{selected_movie}"</b></h3></div>', unsafe_allow_html=True)
        
        # Display recommendations
        cols = st.columns(2)
        for idx, (i, row) in enumerate(recommendations.iterrows()):
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="movie-card">
                    <h3 style="color: #FFD700; border-bottom: 2px solid #FFD700; padding-bottom: 10px;">🎬 {row['title']}</h3>
                    <p><b style="color: #FFD700;">Genre:</b> {row['genres']}</p>
                    <p><b style="color: #FFD700;">Story:</b> {row['description']}</p>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="spotlight-header"><h2>📽️ ALL MOVIES NOW SHOWING</h2></div>', unsafe_allow_html=True)
    
    # Show all movies
    for i, row in movies_df.iterrows():
        with st.expander(f"🎬 {row['title']} - {row['genres']}", expanded=False):
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 20px; border-radius: 10px; border: 1px solid #FFD700;">
                <h4 style="color: #FFD700;">SYNOPSIS</h4>
                <p style="color: white; font-size: 16px;">{row['description']}</p>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="spotlight-header"><h2>🔍 SEARCH MOVIES</h2></div>', unsafe_allow_html=True)
    
    search_term = st.text_input("Enter movie title or keyword:")
    
    if search_term:
        search_results = movies_df[movies_df['title'].str.contains(search_term, case=False) | 
                                  movies_df['description'].str.contains(search_term, case=False)]
        
        if not search_results.empty:
            st.success(f"🎭 Found {len(search_results)} movies!")
            for i, row in search_results.iterrows():
                st.markdown(f"""
                <div class="movie-card">
                    <h4 style="color: #FFD700;">{row['title']}</h4>
                    <p><b>Genre:</b> {row['genres']}</p>
                    <p><b>Description:</b> {row['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("❌ No movies found. Try another search term.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; background: linear-gradient(90deg, #8B0000 0%, #00008B 100%); padding: 20px; border-radius: 10px; border: 2px solid #FFD700;">
    <h3 style="color: #FFD700; margin: 0;">🎭 CINEMAGIC THEATER 🎭</h3>
    <p style="color: white; margin: 5px 0;">Experience the Magic of Cinema • Your Personal Movie Guide</p>
    <p style="color: #FFD700; margin: 0;">Lights • Camera • Action! 🎬</p>
</div>
""", unsafe_allow_html=True)
