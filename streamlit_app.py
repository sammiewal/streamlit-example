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
import spacy             
from string import punctuation
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
tokenizer = ToktokTokenizer()                                                                                   # stopword removal
import pandas as pd
import numpy as np   
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity         
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK stopwords data if not already downloaded
nltk.download("stopwords")
nltk.download("punkt")

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK stopwords and punkt data if not already downloaded
nltk.download("stopwords")
nltk.download("punkt")

from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
    stop_words = set(stopwords.words("english"))
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

combined_df = combined_df[['Repository Name', 'Repository URL', 'Description', 'Keyword', 'Stars']]
combined_df['Description'] = combined_df['Keyword'].map(str) + ' ' + combined_df['Description'].map(str)

# Add in comments
def expand_contractions(text):
    expanded_words = []
    for word in text.split():
        expanded_words.append(contractions.fix(word))
        expanded_text = ' '.join(expanded_words)
    return expanded_text

combined_df["Description"] = combined_df["Description"].apply(lambda x: clean_text(str(x)))


def calculate_word_metrics(text):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.lower() not in punctuation]

    total_words = len(tokens)

    # Handle the case where total_words is zero
    if total_words == 0:
        return total_words, None, None, None

    unique_words = len(set(tokens))
    lexical_div = unique_words / total_words
    avg_word_length = sum(len(word) for word in tokens) / total_words

    return total_words, unique_words, lexical_div, avg_word_length

combined_df[["Total Words", "Unique Words", "Lexical Diversity", "Avg Word Length"]] = combined_df['Description'].apply(calculate_word_metrics).apply(pd.Series)
st.dataframe(combined_df.head())


# add comments
combined_df['word_count'] = combined_df['Description'].apply(lambda x: len(str(x).split())) # splitting up tokens and counting


combined_df['sentence_count'] = combined_df['Description'].apply(lambda x: str(x).count('.') + str(x).count('!') + str(x).count('?'))


combined_df['avg_word_length'] = combined_df['Description'].apply(lambda x: sum(len(word) for word in str(x).split()) / len(str(x).split()) if len(str(x).split()) > 0 else 0)


st.dataframe(combined_df.head())



# Concatenate all descriptions into a single string
all_descriptions = " ".join(combined_df['Description'])

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400).generate(all_descriptions)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

