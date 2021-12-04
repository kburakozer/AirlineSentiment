import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from model import Model
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import spacy
import string
from tensorflow.keras.preprocessing.text import Tokenizer
import re

from model import Model

punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS
# nlp = spacy.load("en_core_web_sm")

# tweets = pd.read_csv('/home/kburakozer/Documents/SWE/SWE/3. Term/SWE599/AirlineSentiment/archive/Tweets.csv')
# news = pd.read_csv('/home/kburakozer/Documents/SWE/SWE/3. Term/SWE599/AirlineSentiment/archive/NewsCategorizer.csv')
# print(news.head())
# print(news.shape)
# print(news["short_description"][0])

path = '/home/kburakozer/Documents/SWE/SWE/3. Term/SWE599/AirlineSentiment/archive/NewsCategorizer.csv'
news = pd.read_csv(path)
# airline_model = Model(path, "text", "airline_sentiment")
# news_model = Model(path, "short_description", "category")
# sentences = tweets[airline_model.values]
# labels = tweets[airline_model.labels]

# print(labels)

# train_sentences, val_sentences, train_labels, val_labels = news_model.data_set()
# print(news_model.CNN())

desc = news["short_description"]
liste = []
for i in desc:
    word_list = desc[i].split(" ")
    for item in word_list:
        liste.append(item)
print(len(liste))
# for i in range(200, 220):
#     print(len(desc[i]))