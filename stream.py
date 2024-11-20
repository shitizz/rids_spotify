#@Music Reccomdation system on streamlit
'''
Authors: 
1. Shitiz Kumar Gupta
2. Keegan Nunes
3. Shrey Agarwal
4. Shashvath Arun  
'''
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import requests
import plotly.graph_objects as go

def get_album_artwork_and_preview(track_name, artist_name):
    """
    Fetch album artwork URL and preview URL from iTunes API.
    Returns a default image URL if artwork is not found.
    """
    try:
        # Construct the query for the iTunes API
        query = f"{track_name} {artist_name}".replace(' ', '+')
        itunes_url = f"https://itunes.apple.com/search?term={query}&media=music&limit=1"

        # Fetch the data
        itunes_response = requests.get(itunes_url)
        itunes_data = itunes_response.json()

        # Check for results in the API response
        if itunes_data.get('resultCount', 0) > 0:
            result = itunes_data['results'][0]

            # Get artwork URL and preview URL
            artwork_url = result.get('artworkUrl100', None)
            if artwork_url:
                artwork_url = artwork_url.replace('100x100', '300x300')  # Resize image
            preview_url = result.get('previewUrl', None)

            return artwork_url or "https://cdn1.iconfinder.com/data/icons/prettyoffice8/256/musicfile.png", preview_url
    except Exception as e:
        st.warning(f"Error fetching data from iTunes: {str(e)}")
    
    # Return fallback values if there are any errors
    return "https://cdn1.iconfinder.com/data/icons/prettyoffice8/256/musicfile.png", None

def load_data():
    # Load the dataset (adjust the path as needed)
    df = pd.read_csv("C:\\Users\\Shitiz\\Desktop\\RIDS_Presentation\\dataset.csv")
    return df

def prepare_features(df):
    # Prepare features for similarity calculation
    feature_cols = [
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'time_signature'
    ]
    X = df[feature_cols]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    genre_dummies = pd.get_dummies(df['track_genre'])
    X_with_genre = np.hstack([X_scaled, genre_dummies.values])
    return X_with_genre

def get_recommendations(df, X_with_genre, song_idx, n_recommendations=5):
    # Calculate cosine similarity
    similarities = cosine_similarity([X_with_genre[song_idx]], X_with_genre)
    selected_song_name = df.iloc[song_idx]['track_name']
    selected_artist_name = df.iloc[song_idx]['artists']

    # Mask to ensure different songs are recommended
    different_names_mask = (df['track_name'] != selected_song_name) | (df['artists'] != selected_artist_name)
    masked_similarities = similarities[0] * different_names_mask

    # Get top N recommendations
    similar_indices = masked_similarities.argsort()[::-1][:n_recommendations]
    recommendations = df.iloc[similar_indices][['track_name', 'artists', 'album_name', 'track_genre']]
    recommendations['similarity_score'] = similarities[0][similar_indices]

    return recommendations

def plot_feature_histogram(df, selected_idx, recommended_indices, features):
    """
    Create a bar plot comparing selected song and recommendations.
    """
    selected_song_features = df.iloc[selected_idx][features]
    recommended_song_features = df.iloc[recommended_indices][features]

    fig = go.Figure()

    # Add bar for the selected song
    fig.add_trace(go.Bar(x=features, y=selected_song_features, name='Selected Song'))

    # Add bars for recommended songs
    for i, idx in enumerate(recommended_indices):
        recommended_features = df.iloc[idx][features]
        fig.add_trace(go.Bar(x=features, y=recommended_features, name=f'Recommendation {i+1}'))

    fig.update_layout(
        barmode='group',
        xaxis_title='Audio Feature',
        yaxis_title='Value',
        legend_title='Song'
    )
    return fig

# Main Streamlit App
def main():
    st.title("Music Recommendation System")
    st.write("Find similar songs based on audio features and genre!")

    # Load and preprocess data
    df = load_data()
    X_with_genre = prepare_features(df)

    # Sidebar for song selection
    st.sidebar.header("Song Selection")
    genres = sorted(df['track_genre'].unique())
    selected_genre = st.sidebar.selectbox("Select a genre:", genres)
    genre_songs = df[df['track_genre'] == selected_genre]
    selected_song = st.sidebar.selectbox("Select a song:", genre_songs['track_name'].tolist())

    # Get selected song index
    song_idx = df[(df['track_name'] == selected_song) & (df['track_genre'] == selected_genre)].index[0]

    # Display selected song details
    st.subheader("Selected Song Details")
    selected_song_info = df.iloc[song_idx][['track_name', 'artists', 'album_name', 'track_genre']]
    artwork_url, preview_url = get_album_artwork_and_preview(selected_song_info['track_name'], selected_song_info['artists'])

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(artwork_url, width=200)
    with col2:
        st.metric("Track", selected_song_info['track_name'])
        st.metric("Artist", selected_song_info['artists'])
        st.metric("Genre", selected_song_info['track_genre'])

    if preview_url:
        st.audio(preview_url, format="audio/mp3")
    else:
        st.write("**Audio preview not available**")

    # Get recommendations
    n_recommendations = st.slider("Number of recommendations:", 1, 15, 5)
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(df, X_with_genre, song_idx, n_recommendations)

        st.subheader("Recommended Songs")
        for _, row in recommendations.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1:
                rec_artwork_url, rec_preview_url = get_album_artwork_and_preview(row['track_name'], row['artists'])
                st.image(rec_artwork_url, width=150)
                if rec_preview_url:
                    st.audio(rec_preview_url, format="audio/mp3")
            with col2:
                st.write(f"**Track:** {row['track_name']}")
                st.write(f"**Artist:** {row['artists']}")
                st.write(f"**Album:** {row['album_name']}")
                st.write(f"**Genre:** {row['track_genre']}")
                st.write(f"**Similarity:** {row['similarity_score']:.2f}")
                st.write("---")

        # Plot feature comparison
        st.subheader("Feature Distribution Comparison")
        features_to_compare = ['danceability', 'energy', 'valence', 'acousticness']
        fig = plot_feature_histogram(df, song_idx, recommendations.index, features_to_compare)
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
