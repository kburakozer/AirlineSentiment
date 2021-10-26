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

punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS
nlp = spacy.load("en_core_web_sm")

tweets = pd.read_csv('/home/kburakozer/Documents/SWE/SWE/3. Term/SWE599/Airline/archive/IMDB Dataset.csv')



def tokenize(words):
    words = words.split(',')
    new_list = []
    new_str = " "
    tokenized_str = " "

    for item in words:
        item = re.sub(r"[0-9]", '', item)
        item = re.sub(r'#\S+', '', item)
        item = re.sub(r'\S@\S+', '', item)
        item = re.sub(r'\S+com', '', item)
        table = item.maketrans("", "", punctuations)
        item = item.translate(table)
        new_list.append(item)
    
    #creating token object
    tokens = nlp(new_str.join(new_list))

    

    #lower, strip and lemmatize
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    
    #remove stopwords, and exclude words less than 2 characters
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations and len(word) > 2]
    return tokenized_str.join(tokens)

max_vocab_length = 10000
max_length = 20

sentences = tweets["review"]
labels = tweets["sentiment"].to_numpy()
le = preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
# labels = np_utils.to_categorical(labels)
train_sentences, val_sentences, train_labels, val_labels = train_test_split(sentences.to_numpy(),
                                                                            labels,
                                                                            test_size=0.2,
                                                                            random_state=42)


for i in range(len(train_sentences)):
    sentence = tokenize(train_sentences[i])
    train_sentences[i] = sentence

# def tokenize(words):
#     words = words.split(',')
#     new_list = []
#     new_str = " "
#     tokenized_str = " "

#     for item in words:
#         item = re.sub(r"[0-9]", '', item)
#         item = re.sub(r'#\S+', '', item)
#         item = re.sub(r'\S@\S+', '', item)
#         item = re.sub(r'\S+com', '', item)
#         table = item.maketrans("", "", punctuations)
#         item = item.translate(table)
#         new_list.append(item)
    

#     tokens = nlp(new_str.join(new_list))

    


#     tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    

#     tokens = [word for word in tokens if word not in stop_words and word not in punctuations and len(word) > 2]
#     return tokenized_str.join(tokens) 

# print(tokenize(train_sentences[0]))



# for i in range(len(train_sentences)):
#     for j in range(len(train_sentences[i]))

# MAX_NUM_WORDS = 100
# print(train_sentences[0])
# tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
# tokenizer.fit_on_texts(train_sentences)
# print(train_sentences[0])
# print(train_sentences[2])
# print(train_labels[2])
text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)

text_vectorizer.adapt(train_sentences)
embedding =  layers.Embedding(input_dim=max_vocab_length,
                                           output_dim=128,
                                           embeddings_initializer="uniform",
                                           input_length=max_length)

inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.LSTM(64, return_sequences=True)(x)
x = layers.LSTM(64)(x)
x = layers.Dense(64, activation="relu")(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model.fit(train_sentences,train_labels, epochs=5)

model_probs = model.predict(val_sentences)
model_probs = tf.squeeze(tf.round(model_probs))
model_accuracy = accuracy_score(val_labels, model_probs) * 100
print(model_accuracy)

def calculate_results(y_true, y_pred):
    model_accuracy = accuracy_score(y_true,y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
            "precision": model_precision,
            "recall": model_recall,
            "f1": model_f1}
    return model_results

results = calculate_results(val_labels, model_probs)
print(results)