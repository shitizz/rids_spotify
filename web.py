import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.figure_factory as ff
import requests
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go


import streamlit as st
import requests

def get_album_artwork(track_name, artist_name):
    """
    Fetch album artwork URL from iTunes API, Spotify Web API, and Last.fm API
    Returns default image URL if artwork not found
    """
    try:
        # Search iTunes API for album artwork
        query = f"{track_name} {artist_name}".replace(' ', '+')
        itunes_url = f"https://itunes.apple.com/search?term={query}&media=music&limit=1"
        itunes_response = requests.get(itunes_url)
        itunes_data = itunes_response.json()

        if itunes_data['resultCount'] > 0:
            # Get the artwork URL and modify it to get larger image
            artwork_url = itunes_data['results'][0]['artworkUrl100']
            artwork_url = artwork_url.replace('100x100', '300x300')
            return artwork_url
    except Exception as e:
        st.warning(f"Could not fetch artwork from iTunes: {str(e)}")

    try:
        # Search Spotify Web API for album artwork
        spotify_headers = {
            'Authorization': 'Bearer <your_spotify_access_token_here>'
        }
        spotify_url = f"https://api.spotify.com/v1/search?q={query}&type=track&limit=1"
        spotify_response = requests.get(spotify_url, headers=spotify_headers)
        spotify_data = spotify_response.json()

        if spotify_data['tracks']['items']:
            spotify_track = spotify_data['tracks']['items'][0]
            album_url = spotify_track['album']['images'][0]['url']
            return album_url
    except Exception as e:
        pass

    try:
        # Search Last.fm API for album artwork
        lastfm_api_key = '<your_lastfm_api_key_here>'
        lastfm_url = f"http://ws.audioscrobbler.com/2.0/?method=track.getinfo&artist={artist_name}&track={track_name}&api_key={lastfm_api_key}&format=json"
        lastfm_response = requests.get(lastfm_url)
        lastfm_data = lastfm_response.json()

        if 'album' in lastfm_data['track']:
            album_url = lastfm_data['track']['album']['image'][-1]['#text']
            return album_url
    except Exception as e:
        st.write(f"***Could not fetch artwork***")

    # Use a default music placeholder image from a reliable source
    return "https://cdn1.iconfinder.com/data/icons/prettyoffice8/256/musicfile.png"


def load_data():
    # In practice, you would load the CSV file
    df = pd.read_csv("C:/Users/keenu/OneDrive/Desktop/RIDS/dataset.csv")
    return df
    
tab1,tab2=st.tabs(["Cosine",'XGBOOST'])

with tab1:
    def prepare_features(df):
        # Select features for similarity calculation
        feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                    'speechiness', 'acousticness', 'instrumentalness', 
                    'liveness', 'valence', 'tempo', 'time_signature']
        
        X = df[feature_cols]
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        genre_dummies = pd.get_dummies(df['track_genre'])
        X_with_genre = np.hstack([X_scaled, genre_dummies.values])
        
        return X_with_genre

    def get_recommendations(df, X_with_genre, song_idx, n_recommendations=5):
        similarities = cosine_similarity([X_with_genre[song_idx]], X_with_genre)
        selected_song_name = df.iloc[song_idx]['track_name']
        selected_artist_name = df.iloc[song_idx]['artists']
        different_names_mask = (df['track_name'] != selected_song_name) | (df['artists'] != selected_artist_name)
        masked_similarities = similarities[0] * different_names_mask
        similar_indices = masked_similarities.argsort()[::-1][:n_recommendations]
        recommendations = df.iloc[similar_indices][['track_name', 'artists', 'album_name', 'track_genre']]
        recommendations['similarity_score'] = similarities[0][similar_indices]
        
        return recommendations

    def plot_feature_histogram(df, selected_idx, recommended_indices, features):
        """
        Create a multiple bar plot comparing the selected song and recommended songs
        for the given audio features.
        
        Parameters:
        df (pandas.DataFrame): The music data DataFrame
        selected_idx (int): The index of the selected song
        recommended_indices (list): The indices of the recommended songs
        features (list): The audio features to plot
        
        Returns:
        plotly.Figure: The multiple bar plot
        """
        selected_song_features = df.iloc[selected_idx][features]
        recommended_song_features = df.iloc[recommended_indices][features]

        fig = go.Figure()

        # Add a bar for the selected song
        fig.add_trace(go.Bar(
            x=features,
            y=selected_song_features,
            name='Selected Song'
        ))

        # Add bars for the recommended songs
        for i, idx in enumerate(recommended_indices):
            recommended_features = df.iloc[idx][features]
            fig.add_trace(go.Bar(
                x=features,
                y=recommended_features,
                name=f'Recommendation {i+1}'
            ))

        fig.update_layout(
            barmode='group',
            xaxis_title='Audio Feature',
            yaxis_title='Value',
            legend_title='Song'
        )

        return fig

    # Modified Streamlit app
    def main():
        st.title("Music Recommendation System")
        st.write("Find similar songs based on audio features and genre!")

        # Load data
        df = load_data()
        X_with_genre = prepare_features(df)

        # Sidebar for selection
        st.sidebar.header("Song Selection")
        
        genres = sorted(df['track_genre'].unique())
        selected_genre = st.sidebar.selectbox(
            "Select a genre:",
            genres,
            index=0
        )
        
        genre_songs = df[df['track_genre'] == selected_genre]
        selected_song = st.sidebar.selectbox(
            "Select a song:",
            genre_songs['track_name'].tolist(),
            index=0
        )

        song_idx = df[(df['track_name'] == selected_song) & 
                    (df['track_genre'] == selected_genre)].index[0]

        # Display selected song info with artwork
        st.subheader("Selected Song Details:")
        selected_song_info = df.iloc[song_idx][['track_name', 'artists', 'album_name', 'track_genre']]
        
        # Create columns for song details and artwork
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Get and display artwork for selected song
            artwork_url = get_album_artwork(selected_song_info['track_name'], 
                                        selected_song_info['artists'])
            st.image(artwork_url, width=200)
        
        with col2:
            
            
                st.metric("Track", selected_song_info['track_name'])
            
                st.metric("Artist", selected_song_info['artists'])
            
                st.metric("Genre", selected_song_info['track_genre'])

        n_recommendations = st.slider("Number of recommendations:", 1, 15, 5)

        if st.button("Get Recommendations"):
            recommendations = get_recommendations(df, X_with_genre, song_idx, n_recommendations)
            
            st.subheader("Recommended Songs:")
            
            # Format similarity scores
            recommendations['similarity_score'] = (recommendations['similarity_score'] * 100).round(2)
            
            # Display recommendations with artwork
            for idx, row in recommendations.iterrows():
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # Get and display artwork for each recommendation
                    artwork_url = get_album_artwork(row['track_name'], row['artists'])
                    st.image(artwork_url, width=150)
                
                with col2:
                    st.markdown(f"**Track:** {row['track_name']}")
                    st.markdown(f"**Artist:** {row['artists']}")
                    st.markdown(f"**Album:** {row['album_name']}")
                    st.markdown(f"**Genre:** {row['track_genre']}")
                    st.markdown(f"**Similarity:** {row['similarity_score']:.2f}%")
                    st.markdown("---")

            # Display feature comparison
            st.subheader("Feature Distribution Comparison")
            features_to_compare = ['danceability', 'energy', 'valence', 'acousticness']
            
            feature_descriptions = {
                'danceability': 'How suitable a track is for dancing (0.0 to 1.0)',
                'energy': 'Intensity and activity measure (0.0 to 1.0)',
                'valence': 'Musical positiveness measure (0.0 to 1.0)',
                'acousticness': 'How likely a track is to be acoustic (0.0 to 1.0)'
            }
            
            with st.expander("Feature Descriptions"):
                for feature, desc in feature_descriptions.items():
                    st.write(f"**{feature}**: {desc}")
            
            fig = plot_feature_histogram(
                df, 
                song_idx, 
                recommendations.index,
                features_to_compare
            )
            
            st.plotly_chart(fig)

            # Display audio features of selected song
            st.subheader("Audio Features of Selected Song")
            features_df = df.iloc[song_idx][features_to_compare]
            
            feature_cols = st.columns(len(features_to_compare))
            for i, (feature, value) in enumerate(features_df.items()):
                if feature == 'tempo':
                    normalized_value = min(value / 200, 1)
                    feature_cols[i].metric(
                        feature.title(),
                        f"{value:.1f} BPM",
                        delta=None
                    )
                else:
                    normalized_value = value
                    feature_cols[i].metric(
                        feature.title(),
                        f"{value:.3f}",
                        delta=None
                    )

    if __name__ == "__main__":
        main()