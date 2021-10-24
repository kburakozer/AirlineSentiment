import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class Model:
    max_vocab_length = 10000
    max_length = 20
    epochs = 1

    def data_set(self):
        
        # read tweets and tags csv files
        tweets = pd.read_csv('/home/kburakozer/Documents/SWE/SWE/3. Term/SWE599/Airline/archive/Tweets.csv')

        sentences = tweets["text"]
        labels = tweets["airline_sentiment"]
        labels = tweets["airline_sentiment"].to_numpy()
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        labels = np_utils.to_categorical(labels)
        train_sentences, val_sentences, train_labels, val_labels = train_test_split(sentences.to_numpy(),
                                                                                    labels,
                                                                                    test_size=0.2,
                                                                                    random_state=42)

        return train_sentences, val_sentences, train_labels, val_labels


    def vectorizer(self):       
        text_vectorizer = TextVectorization(max_tokens=self.max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=self.max_length)

        train_sentences, val_sentences, train_labels, val_labels = self.data_set()
        text_vectorizer.adapt(train_sentences)

        return text_vectorizer

    def embedding(self):
        embedding_layer = layers.Embedding(input_dim=self.max_vocab_length,
                                           output_dim=128,
                                           embeddings_initializer="uniform",
                                           input_length=self.max_length)
        return embedding_layer


    def bayes_model(self):
        train_sentences, val_sentences, train_labels, val_labels = self.data_set()
        model = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", MultinomialNB())
        ])

        model.fit(train_sentences, train_labels)
        score = model.score(val_sentences, val_labels)

        return score

    def calculate_results(self,y_true, y_pred):
        model_accuracy = accuracy_score(y_true,y_pred) * 100
        model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        model_results = {"accuracy": model_accuracy,
                "precision": model_precision,
                "recall": model_recall,
                "f1": model_f1}
        return model_results

    def LSTM(self):
        train_sentences, val_sentences, train_labels, val_labels = self.data_set()
        inputs = layers.Input(shape=(1,), dtype="string")
        embedding = self.embedding()
        text_vectorizer = self.vectorizer()
        x = text_vectorizer(inputs)
        x = embedding(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LSTM(64)(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(3, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=["accuracy"])
        model.fit(train_sentences,train_labels, epochs=self.epochs)

        model_probs = model.predict(val_sentences)
        model_probs = tf.squeeze(tf.round(model_probs))
        results = self.calculate_results(val_labels, model_probs)
        return results


