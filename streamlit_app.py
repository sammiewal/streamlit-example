import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the dataset
data = pd.read_csv("/content/training.1600000.processed.noemoticon.csv", encoding='ISO-8859-1', header=None)
data.columns = ["sentiment", "id", "date", "flag", "user", "text"]

def clean_text(tweet):
    # Decoding HTML
    tweet = BeautifulSoup(tweet, "lxml").get_text()
    # Convert to lowercase
    tweet = tweet.lower()
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    # Remove mentions
    tweet = re.sub(r'@\S+', '', tweet)
    # Remove hashtags (just the # symbol)
    tweet = re.sub(r'#', '', tweet)
    # Remove punctuations
    tweet = re.sub(r'[^\w\s]', '', tweet)

    # Text of characters to exclude
    exclude_text = "中英文敏感词语言检测中外手机电话归属地运营商查询名字推断性别手机号抽取身份证抽取邮箱抽取中日"

    # Check if the tweet contains any character from exclude_text
    if any(char in tweet for char in exclude_text):
        return ''

    # Tokenization
    tokens = nltk.word_tokenize(tweet)

    # Remove stopwords (optional based on performance)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization (optional based on performance)
    # lemmatizer = WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into a string
    cleaned_tweet = ' '.join(tokens)

    return cleaned_tweet
# Apply the cleaning function to the text column
data["cleaned_text"] = data["text"].apply(clean_text)
