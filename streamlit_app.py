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


from wordcloud import WordCloud
import matplotlib.pyplot as plt

from PIL import Image

@st.cache_data
def load_data(url):
    data = pd.read_csv(url)
    return data

st.title('Repository Refiner')
st.write(
    'Dive into the GitHub Repository Refiner and start exploring the interconnected '
    'world of open-source projects like never before. Whether you\'re looking to '
    'contribute, collaborate, or simply satisfy your curiosity, our tool is here to '
    'refine your search and discovery process on GitHub.'
)

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


# add comments
combined_df['word_count'] = combined_df['Description'].apply(lambda x: len(str(x).split())) # splitting up tokens and counting

combined_df['avg_word_length'] = combined_df['Description'].apply(lambda x: sum(len(word) for word in str(x).split()) / len(str(x).split()) if len(str(x).split()) > 0 else 0)



# Concatenate all descriptions into a single string
all_descriptions = " ".join(combined_df['Description'])


stopwords = set(nltk_stopwords.words('english'))

new_stopwords = ['e', 'c', 'using']

new_stopwords_list =  stopwords.union(new_stopwords)

print(new_stopwords_list)

image_path = 'github.png'

mask_image = Image.open(image_path)

# Convert the image to 'L' mode which gives you grayscale
mask_image_gray = mask_image.convert('L')

# Create a binary mask where white is 255 and black is 0
mask_array = np.array(mask_image_gray)
transformed_mask_image = np.where(mask_array == 255, 1, 0)


# Load the image file
mask_image_color = Image.open(image_path)

# Convert it to grayscale based on the alpha channel
mask_image_gray = mask_image_color.split()[-1]

# Create a binary mask where white is 255 and black is 0
mask_array = np.array(mask_image_gray)
transformed_mask_image = np.where(mask_array == 0, 0, 255)  # Inverting the mask if necessary

# Define a new transformation function for clarity
def transform_format(val):
    if val == 0:
        return 0  # Black
    else:
        return 255  # White

# Apply the transformation to each pixel in the mask
for i in range(len(transformed_mask_image)):
    transformed_mask_image[i] = list(map(transform_format, transformed_mask_image[i]))


# Convert it to grayscale
mask_image_gray = mask_image_color.convert('L')

# Create a binary mask where the guitar outline is white (255) and the background is black (0)
# Assuming the guitar outline is currently black and the background is white
mask_array = np.array(mask_image_gray)
transformed_mask_image = np.where(mask_array < 128, 0, 255)  # Adjust threshold as necessary

st.title('Text Exploration')
st.write('Analyze trending topics amongst the GitHub community.')

# Create the word cloud object with additional parameters
wordcloud = WordCloud(
    background_color='black',
    mask=transformed_mask_image,
    width=2000,
    height=2000,
    max_font_size=300,  # Adjust the maximum font size if necessary
    stopwords=new_stopwords_list,
    contour_color='steelblue',
    contour_width=2
).generate(all_descriptions)

fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

st.title('Topic Modeling')

preprocessed_data = combined_df['Description']
tv = TfidfVectorizer(min_df=3, max_df=0.7, ngram_range=(2,2))
dtm = tv.fit_transform(preprocessed_data)
vocabulary = np.array(tv.get_feature_names_out())
# Fit LDA Model
lda = LatentDirichletAllocation(n_components=4, random_state=42)
lda.fit(dtm)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


no_top_words = 10
display_topics(lda, vocabulary, no_top_words)

def custom_tokenizer(text):
    tokens = nltk.word_tokenize(text)
    english_stopwords = set(nltk_stopwords.words('english')) 
    tokens = [token for token in tokens if token not in english_stopwords]
    bigrams = ["_".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    return bigrams

# Assuming preprocessed_data is defined and accessible
tokenized_texts = [custom_tokenizer(text) for text in preprocessed_data]

gensim_dict = Dictionary(tokenized_texts)
gensim_dict.filter_extremes(no_below=2, no_above=0.70) # we need to set this to the same filtering we did during our vectorization step
print(gensim_dict)

lda = LatentDirichletAllocation(n_components=4, random_state=42)
doc_topic = lda.fit(dtm)

no_top_words = 10
display_topics(doc_topic, vocabulary, no_top_words)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

def prepare_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d" % (topic_idx)] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)

st.write('Uncover latent topics within the GitHub repositories.')

# Fit the LDA model
lda = LatentDirichletAllocation(n_components=4, random_state=42)
lda.fit(dtm)

# Get the feature names (vocabulary)
feature_names = vocabulary  # Replace 'vocabulary' with your actual vocabulary variable

# Prepare the topics data for visualization
topics_df = prepare_topics(lda, feature_names, no_top_words)

# New topic names
new_topic_names = {
    0: "Open-Source Web Development",
    1: "Machine Learning and Development Standards",
    2: "Cloud Computing and Open-Source Frameworks",
    3: "Artificial Intelligence and Developer Resources",
}

# Modified display_topics function to use new topic names
def display_topics(model, feature_names, no_top_words, topic_names):
    for topic_idx, topic in enumerate(model.components_):
        print(f"{topic_names.get(topic_idx, 'Topic ' + str(topic_idx))}:")
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

# Prepare the topics data for visualization with new topic names
def prepare_topics(model, feature_names, no_top_words, topic_names):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_key = topic_names.get(topic_idx, 'Topic ' + str(topic_idx))
        topic_dict[topic_key] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)

# Use the modified display_topics function
display_topics(lda, feature_names, no_top_words, new_topic_names)

# Prepare the topics data for visualization
topics_df = prepare_topics(lda, feature_names, no_top_words, new_topic_names)

doc_topic_matrix = lda.transform(dtm)

df_doc_topic = pd.DataFrame(doc_topic_matrix, columns=[f'Topic {i}' for i in range(lda.n_components)])

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
km = KMeans(n_clusters=5,
            max_iter=10000,
            n_init = 50,
            random_state = 42
            ).fit(tv_matrix) # fit kmeans
Counter(km.labels_) # find count of cluster labels
# rerun kmeans with new cluster amount
# apply counter

cosine_sim_features = cosine_similarity(tv_matrix)# get cosine similarity features from tv_matrix

tv = TfidfVectorizer(use_idf=True, min_df=3, max_df=0.8, ngram_range=(1,2), sublinear_tf=True)
tv_matrix = tv.fit_transform(norm_corpus)

                         # set parameters for tf-idf for unigrams and bigrams
tfidf_matrix = tv.fit_transform(norm_corpus)                                      # extract tfidf features from norm_corpus

doc_sim = cosine_similarity(tfidf_matrix)    # compute document similarity by examining the cosine similairty b/w documents in matrix
doc_sim_df = pd.DataFrame(doc_sim)                                                  # take doc_sim, convert to dataframe
doc_sim_df.head()

# Step 1: Assign each document to its most dominant topic
dominant_topic = np.argmax(doc_topic_matrix, axis=1)  # This gives the index of the most dominant topic for each document

# Step 2: Create a mapping function
def map_topic(index):
    return new_topic_names.get(index, f"Topic {index}")

# Step 3: Map each document to its topic name and create the 'Topics' column
combined_df['Topics'] = [map_topic(idx) for idx in dominant_topic]

# Now you can use this 'Topics' column in your Streamlit app


# Step 2: Create a multiselect sidebar for topics
selected_topics = st.sidebar.multiselect(
    'Select Topics', 
    options=combined_df['Topics'].unique(),  # List of unique topics
    default=combined_df['Topics'].unique()[0]  # Default selection
)

# Step 3: Filter the DataFrame based on the selected topics
filtered_df = combined_df[combined_df['Topics'].isin(selected_topics)]

# Step 4: Display the filtered repositories
for index, row in filtered_df.iterrows():
    st.write(f"Repository Name: {row['Repository Name']}")
    st.write(f"URL: {row['Repository URL']}")
    st.write(f"Description: {row['Description']}")
    st.write(f"Stars: {row['Stars']}")
    st.write(f"Topics: {row['Topics']}")
    st.write("-----")




# saving all the unique movie titles to a list
repository_list = combined_df['Repository Name'].values

stop_words = nltk.corpus.stopwords.words('english')

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

