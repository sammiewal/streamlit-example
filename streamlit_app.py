import streamlit as st
from text_exploration import *
import streamlit as st
import os
import requests
import json
import pandas as pd
import nltk
from urllib import request
from bs4 import BeautifulSoup                                                                                   # needed for parsing HTML
import contractions                                                                                             # contractions dictionary
from string import punctuation
import seaborn as sns
import spacy             
from string import punctuation
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.cluster import AffinityPropagation
from nltk.corpus import stopwords as nltk_stopwords 
tokenizer = ToktokTokenizer()                                                                                   # stopword removal
import pandas as pd
import numpy as np 
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity         
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import gensim

# Download NLTK stopwords data if not already downloaded
nltk.download("stopwords")
nltk.download("punkt")

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


@st.cache_data
def load_data(url):
    data = pd.read_csv(url)
    return data

# URL of the raw CSV file
csv_url = 'https://raw.githubusercontent.com/sammiewal/streamlit-example/master/combined_data.csv'

# Load the data
combined_df = load_data(csv_url)

# Define a function to filter out specific characters
def filter_characters(text):
    characters_to_filter = ["通", "义", "千", "问", "书", "生"]
    return ''.join([char for char in text if char not in characters_to_filter])

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

def remove_special_characters(text):
    if isinstance(text, str):  # Check if the input is a string
        return re.sub(r'[^a-zA-Z0-9 _]', '', text)
    else:
        return ''  # Return an empty string or some default value for non-string inputs

# Define a function to clean text with lemmatization
def clean_text(text):
    # Handle non-string inputs
    if not isinstance(text, str):
        return ''

    exclude_text = "中英文敏感词语言检测中外手机电话归属地运营商查询名字推断性别手机号抽取身份证抽取邮箱抽取中日"

    if any(char in text for char in exclude_text):
        return ''

    # Remove URLs, mentions, hashtags, and non-alphanumeric characters in one step
    text = re.sub(r'http\S+|@\S+|#|[^\w\s]', '', text)

    # Tokenize, lowercase, and remove non-alphanumeric words
    words = word_tokenize(text)
    cleaned_words = [word.lower() for word in words if word.isalnum()]

    # Remove stopwords
    stop_words = set(nltk_stopwords.words("english"))
    cleaned_words = [word for word in cleaned_words if word not in stop_words]

    # Lemmatize
    lemmatized_words = [lemmatizer.lemmatize(word) for word in cleaned_words]

    # Special handling for specific cases (e.g., "pdf")
    lemmatized_words = ['pdf' if word.startswith('pdf') else word for word in lemmatized_words]

    return " ".join(lemmatized_words)

# Load the data and make a copy
combined_df = load_data(csv_url).copy()

# Apply the cleaning function to the copy of your DataFrame
combined_df["Description"] = combined_df["Description"].apply(remove_special_characters)
combined_df["Description"] = combined_df["Description"].apply(clean_text)

stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)                                       # lower case and remove special characters\whitespaces
    doc = doc.lower()
    doc = doc.strip()
    tokens = nltk.word_tokenize(doc)                                                          # tokenize document
    filtered_tokens = [token for token in tokens if token not in stop_words]                  # filter stopwords out of document
    doc = ' '.join(filtered_tokens)                                                           # re-create document from filtered tokens
    return doc

normalize_corpus = np.vectorize(normalize_document)

norm_corpus = normalize_corpus(list(combined_df['Description']))

tv = TfidfVectorizer(use_idf=True, min_df=3, max_df=0.8, ngram_range=(1,2), sublinear_tf=True)
tv_matrix = tv.fit_transform(norm_corpus)

                         # set parameters for tf-idf for unigrams and bigrams
tfidf_matrix = tv.fit_transform(norm_corpus)                                      # extract tfidf features from norm_corpus

doc_sim = cosine_similarity(tfidf_matrix)    # compute document similarity by examining the cosine similairty b/w documents in matrix
doc_sim_df = pd.DataFrame(doc_sim)                                                  # take doc_sim, convert to dataframe
doc_sim_df.head()

# saving all the unique movie titles to a list
repository_list = combined_df['Repository Name'].values

# Create a Streamlit app
st.title('Repository Recommender System')

# Input field for search query
search_query = st.text_input('Enter a search query:')

# Function to find similar repositories based on the search query
def query_repository_recommender(search_query, repository_list, tfidf_matrix, tv, combined_df):
    try:
        # Transform the search query into its vector form
        query_vector = tv.transform([search_query])

        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)

        similar_repository_idxs = cosine_similarities[0].argsort()[-5:][::-1]

        similar_repositories = repository_list[similar_repository_idxs]

        return similar_repositories
    except Exception as e:
        return ["Error: " + str(e)]

# Button to trigger the search and display recommendations
if st.button('Find Similar Repositories'):
    query_recommendations = query_repository_recommender(search_query, repository_list, tfidf_matrix, tv, combined_df)
    
    if "Error" in query_recommendations[0]:
        st.write("An error occurred:", query_recommendations[0])
    else:
        st.write("Based on your search query, I'd recommend checking out:")
        for repo in query_recommendations:
            # Retrieve the corresponding repository URL from the DataFrame
            repo_url = combined_df.loc[combined_df['Repository Name'] == repo, 'Repository URL'].values[0]
            
            # Display the repository name as a clickable hyperlink
            st.markdown(f"[{repo}]({repo_url})")

