**🎵 Music Recommendation System**
This project showcases the analysis of Spotify music data and the development of a personalized Music Recommendation System. Leveraging machine learning techniques and interactive web deployment, the project aims to enhance the music discovery experience by suggesting songs tailored to user preferences.

**📂 Project Structure**
Here’s an overview of the repository files and their purposes:

`Music_EDA.ipynb`

• This Jupyter Notebook contains the Exploratory Data Analysis (EDA) performed on the Spotify dataset.
• Key analyses include:
  ‣  Popularity trends among genres and artists.
  ‣  Relationships between audio features like energy, loudness, and tempo.
  ‣  Visualization of insights using libraries like matplotlib and seaborn.
• The notebook also preprocesses the dataset for the recommendation system.

`stream.py`

• The main application script, built with Streamlit, for deploying the recommendation system.
• Features:
  ‣  A sidebar for selecting genres and songs.
  ‣  Integration with the iTunes API for fetching album artwork and audio previews.
  ‣  Display of recommended songs with detailed metadata and similarity scores.
  ‣  Interactive visualizations comparing song features.
  
`Music Analysis PPT.pptx`

• A presentation that outlines:

  ‣  The project’s objectives, methodology, and key insights.
  
  ‣  Graphs and charts derived from the EDA.
  
  ‣  Summary of the recommendation system’s design and future scope.
  
**🎯 Objectives**

The goal of this project is to:

• Analyze the attributes of songs to uncover trends and patterns in music listening.

• Develop a hybrid recommendation system combining collaborative filtering and content-based techniques.

• Enhance the user experience by recommending songs that balance familiarity with novelty.

**🚀 Features**

• Data Analysis 

➼ EDA revealed trends like:

    ➼ Top Artists and Genres: Quevedo, Bizarrap, and Pop-film are highly popular.
    
    ➼ Energy-Loudness Correlation: A positive correlation was observed, indicating higher energy tracks are generally louder.
    
    ➼ Popularity Distribution: Mainstream and niche artists show distinct trends.
    
•  Recommendation System

    • Hybrid Model: Combines collaborative filtering (user and item-based) with content-based filtering.
    
    •  Cosine Similarity: Used to calculate similarity between songs based on vectorized features.
    
• Interactive App: Streamlit-based interface for exploring recommendations with visual and audio insights.


**📊 Dataset**

Source: Spotify dataset with metadata for thousands of songs.

Attributes: Includes audio features such as tempo, energy, loudness, valence, danceability, and more.


**🛠️ Tools and Technologies**

Python: Core programming language for analysis and development.

Libraries:

Data Analysis: pandas, numpy

Visualizations: matplotlib, seaborn, plotly

Machine Learning: scikit-learn

Web App: streamlit

APIs: iTunes API for fetching album artwork and audio previews.


**📝 How to Run**

*Prerequisites*

Python 3.x

Required libraries (install via pip install -r requirements.txt):

`pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
plotly
requests`

**Notes**

• Ensure the dataset is available at the specified path in the code (C:\\Users\\Shitiz\\Desktop\\RIDS_Presentation\\dataset.csv or update the path).

• The iTunes API integration requires an active internet connection.

**🔑 Key Insights**

The most popular song in the dataset is “Unholy” with a popularity score of 100.

Genres like Pop-film and K-pop dominate in average popularity, while romance is the least popular.

Happy songs (mode 1) generally have higher energy levels than sad songs (mode 0).

**🌐 Future Scope**

Emotion Analysis: Integrate mood detection to suggest songs matching a listener's emotional state.

Cross-Platform Recommendations: Expand compatibility with other streaming platforms.

Mental Health Applications: Use the recommendation system in wellness or therapeutic settings.


**🙏 Acknowledgments**

Team Members:

Keegan Nunes

Shrey Agarwal

Shashvath Arun

Mentor: Dr. Kavita Jain for her invaluable guidance.
