import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
from tensorflow.keras.regularizers import Regularizer
from tensorflow.python.keras.regularizers import L1
import matplotlib.pyplot as plt

punctuations = string.punctuation
nlp = spacy.load("en_core_web_sm")
stop_words = spacy.lang.en.stop_words.STOP_WORDS


class Model:
    max_vocab_length = 10000
    max_length = 20
    epochs = 5
    path = ""
    values = ""
    labels = ""
    category_num = 10

    def __init__(self, path, values, labels):
        self.path = path
        self.values = values
        self.labels = labels

    def data_set(self):
        
        # read tweets and tags csv files
        tweets = pd.read_csv(self.path)


        sentences = tweets[self.values]
        labels = tweets[self.labels]
        labels = tweets[self.labels].to_numpy()
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        labels = np_utils.to_categorical(labels)
        train_sentences, val_sentences, train_labels, val_labels = train_test_split(sentences.to_numpy(),
                                                                                    labels,
                                                                                    test_size=0.2,
                                                                                    random_state=42)
        for i in range(len(train_sentences)):
            sentence = self.tokenize(train_sentences[i])
            train_sentences[i] = sentence
        for i in range(len(val_sentences)):
            sentence = self.tokenize(val_sentences[i])
            val_sentences[i] = sentence
        return train_sentences, val_sentences, train_labels, val_labels

    def tokenize(self, words):
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
        # tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
        
        #remove stopwords, and exclude words less than 2 characters
        # tokens = [word for word in tokens if word not in stop_words and word not in punctuations and len(word) > 2]
        return tokenized_str.join(tokens) 


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

    def k_means(self):
        train_sentences, val_sentences, train_labels, val_labels = self.data_set()
        model = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", KMeans(n_clusters=3))
        ])
        model = model.fit(train_sentences, train_labels)
        predictions = model.predict(val_sentences)
        # return self.calculate_results(val_labels,predictions)
        # return predictions
        # df = pd.DataFrame(model)
        # filtered_label0 = df[labels == 0]
        
        # plotting the results
        # plt.scatter(filtered_label0[:,0] , filtered_label0[:,1])
        # plt.scatter(df)
        # plt.show()
        


    def calculate_results(self,y_true, y_pred):
        model_accuracy = accuracy_score(y_true,y_pred) * 100
        model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        model_results = {"accuracy": model_accuracy,
                "precision": model_precision,
                "recall": model_recall,
                "f1": model_f1}
        return model_results

    def plot_loss_curves(self, history):
        """
        Returns separate loss curves for training and validation metrics.
        """ 
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        epochs = range(len(history.history['loss']))

        # Plot loss
        plt.plot(epochs, loss, label='training_loss')
        plt.plot(epochs, val_loss, label='val_loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.rc('font', size=18)          # controls default text sizes
        plt.rc('axes', titlesize=18)     # fontsize of the axes title
        plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
        plt.rc('legend', fontsize=18)    # legend fontsize
        plt.rc('figure', titlesize=18)  # fontsize of the figure title

        plt.legend()

        # Plot accuracy
        plt.figure()
        plt.plot(epochs, accuracy, label='training_accuracy')
        plt.plot(epochs, val_accuracy, label='val_accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.rc('font', size=18)          # controls default text sizes
        plt.rc('axes', titlesize=18)     # fontsize of the axes title
        plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
        plt.rc('legend', fontsize=18)    # legend fontsize
        plt.rc('figure', titlesize=18)  # fontsize of the figure title
        plt.legend()

        plt.show()

    def LSTM(self):
        train_sentences, val_sentences, train_labels, val_labels = self.data_set()
        inputs = layers.Input(shape=(1,), dtype="string")
        embedding = self.embedding()
        text_vectorizer = self.vectorizer()
        x = text_vectorizer(inputs)
        x = embedding(x)
        x = layers.LSTM(32, return_sequences=True,activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(16, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        # x = layers.Dropout(0.2)(x)
        # x = layers.LSTM(64,return_sequences=True)(x)
        # x = layers.LSTM(128)(x)
        # x = layers.Dropout(0.2)(x)
        # x = layers.Dense(64, activity_regularizer=L1(0.01), activation="relu")(x)
  
        # x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
        # x = layers.Dense(16, activation='relu')(x)
        outputs = layers.Dense(self.category_num, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=["accuracy"])
        history = model.fit(train_sentences,
                            train_labels, 
                            validation_data= (val_sentences, val_labels),
                            epochs=self.epochs)

        self.plot_loss_curves(history)
        model_probs = model.predict(val_sentences)
        model_probs = tf.squeeze(tf.round(model_probs))
        results = self.calculate_results(val_labels, model_probs)
        return results

    def CNN(self):
        embedding = self.embedding()
        text_vectorizer = self.vectorizer()
        train_sentences, val_sentences, train_labels, val_labels = self.data_set()
        inputs = layers.Input(shape=(1,), dtype="string")

        x = text_vectorizer(inputs)
        x = embedding(x)
        x = layers.Conv1D(filters=20,
                          kernel_size=3,
                          activation="relu",
                          padding="valid")(x)
        x = layers.Conv1D(8, 3, activation="relu")(x)
        # x = layers.Conv1D(8, 2, activation="relu")(x)
        x = layers.GlobalMaxPooling1D()(x)
        outputs = layers.Dense(self.category_num, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=["accuracy"])
        history = model.fit(train_sentences,
                            train_labels, 
                            validation_data= (val_sentences, val_labels),
                            epochs=self.epochs)

        self.plot_loss_curves(history)
        model_probs = model.predict(val_sentences)
        model_probs = tf.squeeze(tf.round(model_probs))
        results = self.calculate_results(val_labels, model_probs)
        return results

