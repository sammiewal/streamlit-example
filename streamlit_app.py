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
import spacy                                                                                                    # used for lemmatization/stemming
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
tokenizer = ToktokTokenizer()                                                                                   # stopword removal
from nltk import word_tokenize
import pandas as pd
import numpy as np   

# general packages for data manipulation
TOKEN = st.secrets["GITHUB_KEY"]

# Define the API endpoint
endpoint = "https://api.github.com/search/repositories"

# Keywords you want to search for
keywords = ['api', 'tensorflow', 'machine learning', 'python', 'ajax' ,'cloud computing', 'twitter', 'chatgpt', 'pytorch', 'keras', 'ai', 'django', 'ruby', 'ohmyzsh', 'arduino', 'chrome', 'wordpress', 'vinta', 'development', 'twbs', 'bootstrap', 'cybersecurity', 'docker', 'public api', 'aws', 'ibm', 'bitcoin', 'microsoft', 'drupal', 'matplotlib', 'seaborn', 'freecodecamp', 'react', 'donnemartin', 'html5', 'android',  'typescript', 'kamranahmedse', 'azure', 'atom', 'mysql', 'database', 'algorithm', 'nextjs', 'sass', 'sindresorhus', 'css', 'config', 'nltk', 'spacy', 'java', 'nodejs', 'javascript', 'scikit learn', 'jwasham',  'ebookfoundation', 'jquery', 'angular']

# Set up the headers
headers = {
    "Authorization": f"token {TOKEN}"
}

# Initialize an empty list to store DataFrames for each keyword
dfs = []

# Initialize an empty list to store the JSON response data
json_responses = []

for keyword in keywords:
    # Define the query parameters for the current keyword and language filter
    params = {
        "q": f"{keyword}",  # Keyword and language filter
        "sort": "stars",  # Sort by stars (you can change this to other criteria)
        "order": "desc"   # Order by descending (you can change to "asc" for ascending)
    }

    # Send a GET request to the API endpoint
    response = requests.get(endpoint, headers=headers, params=params)

    # Check the response status code
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        repositories = data["items"]

        # Create a DataFrame from the repository data
        df = pd.DataFrame(repositories)

        # Select and rename columns of interest
        df = df[["name", "html_url", "description", "stargazers_count", "forks_count", "watchers_count"]]
        df.columns = ["Repository Name", "Repository URL", "Description", "Stars", "Forks", "Watchers"]

        # Add a new column for the keyword
        df["Keyword"] = keyword

        # Append the DataFrame to the list
        dfs.append(df)

        # Append the JSON response to the list
        json_responses.append(data)
    else:
        st.write(f"Failed to retrieve repositories for '{keyword}'. Status code: {response.status_code}")

# Combine DataFrames for each keyword into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Display the combined DataFrame
st.write(combined_df)


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK stopwords data if not already downloaded
nltk.download("stopwords")
nltk.download("punkt")

# Define a function to filter out specific characters
def filter_characters(text):
    # Define a list of characters to filter out
    characters_to_filter = ["通", "义", "千", "问", "书", "生"]

    # Replace the characters with an empty string
    filtered_text = ''.join([char for char in text if char not in characters_to_filter])

    return filtered_text

# Apply character filtering to the "Description" column
combined_df["Description"] = combined_df["Description"].apply(lambda x: filter_characters(str(x)))

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function to clean text with lemmatization
def clean_text(text):
    # Tokenize the text (split into words)
    words = word_tokenize(text)

    # Remove punctuation and convert to lowercase
    cleaned_words = [word.lower() for word in words if word.isalnum()]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    cleaned_words = [word for word in cleaned_words if word not in stop_words]

    # Lemmatize the words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in cleaned_words]

    # Join the lemmatized words back into text
    cleaned_text = " ".join(lemmatized_words)

    return cleaned_text

# Apply text cleaning with lemmatization to the "Description" column
combined_df["Description"] = combined_df["Description"].apply(lambda x: clean_text(str(x)))

# Example usage
st.write(combined_df.head())
combined_df.head()
