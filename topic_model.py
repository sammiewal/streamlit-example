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

