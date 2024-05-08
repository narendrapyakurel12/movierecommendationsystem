import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
@st.cache_resource
def load_model():
    movies_list = pd.read_pickle("movies.pkl")
    return movies_list
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(load_model()['title'] + ' ' + load_model()['cast'] + ' ' + load_model()['listed_in']+' ' + load_model()['director']+' ' + load_model()['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
def recommend_movies(movie_title, top_n=10):
    idx = load_model()[load_model()['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return load_model().iloc[movie_indices].title
st.title('Movie Recommendation System')

query_movie_title = st.selectbox('Select a movie title', load_model()['title'], index=0)

#Button to trigger recommendation
if st.button('Get Recommendations'):
    recommended_movies = recommend_movies(query_movie_title)
    st.write('Recommended Movies:')
    for movie in recommended_movies:
        st.write(movie)

           
