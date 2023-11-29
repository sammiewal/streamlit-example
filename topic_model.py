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
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
tokenizer = ToktokTokenizer()                                                                                   # stopword removal
import numpy as np   
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity         
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK stopwords data if not already downloaded
nltk.download("stopwords")
nltk.download("punkt")

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import gensim
import matplotlib.pyplot as plt

@st.cache_data
def load_data(url):
    data = pd.read_csv(url)
    return data

def app():
    st.title('Topic Modeling')
    st.write('This is the main page of the app.')
    # URL of the raw CSV file
    csv_url = 'https://raw.githubusercontent.com/sammiewal/streamlit-example/master/combined_data.csv'
    # Load the data
    combined_df = load_data(csv_url)
    preprocessed_data = combined_df['Description']
    tv = TfidfVectorizer(min_df=3, max_df=0.7, ngram_range=(2,2))
    dtm = tv.fit_transform(preprocessed_data)
    vocabulary = np.array(tv.get_feature_names_out())
    # Fit LDA Model
    lda = LatentDirichletAllocation(n_components=4, random_state=42)
    lda.fit(dtm)
    no_top_words = 10
    display_topics(doc_topic, vocabulary, no_top_words)

    def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    import pandas as pd

def prepare_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d" % (topic_idx)] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        topic = lda.components_[i]
        top_word_indices = topic.argsort()[-no_top_words:]
        ax.barh(range(no_top_words), topic[top_word_indices])
        ax.set_yticks(range(no_top_words))
        ax.set_yticklabels([feature_names[j] for j in top_word_indices])
        ax.set_title('Topic %d' % i)

    plt.tight_layout()
    plt.show()

    

