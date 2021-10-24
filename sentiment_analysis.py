import pandas as pd
import matplotlib.pyplot as plt
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier

# list of stopwprds
stop_words = stopwords.words("english")

# function that cleans the tweets from stopwords, punctuations, urls, duplicate words, symbols
def preprocess(tweet):
    tweet.lower()
    tweet = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                  '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet)
    tweet = re.sub("(@[A-Za-z0-9_]+)", "", tweet)

    # deleting punctuations
    table = tweet.maketrans("", "", string.punctuation)
    tweet2 = tweet.translate(table)

    # tokenization process
    tokens = word_tokenize(tweet2)
    filtered_tokens = []
    for item in tokens:
        if item not in stop_words:
            filtered_tokens.append(item)

    # list of lemmetized words
    lemmatized_words = []
    lemmatizer = WordNetLemmatizer()

    for item in filtered_tokens:
        lemmatized_words.append(lemmatizer.lemmatize(item))

    # joining tokens into string
    return (" ".join(filtered_tokens)).lower()

# function for vectorizing the tweets
def vectorizer(tweet):
    # as training data included 1844 features, the whole data is limited to 1844 by max_feature parameter.
    vectorizer = CountVectorizer(max_features=1844)
    vectorizer.fit(tweet)
    vector = vectorizer.transform(tweet)
    return vector.toarray()


# reading the tweets which were manually labeled
game_data = "archive/tag_englishTweets.csv"
game_read = pd.read_csv(game_data)

# converting the data into tweet list
game_tweets = game_read.text.to_numpy()
# converting the data into labels for the tweets
game_polarity = game_read.polarity.to_numpy()

# training data
training = game_tweets[:449]
label = game_polarity[:449]

# list to hold the cleaned data
game_pre_processed_data = []

# process of cleaning each tweet in the training set
for item in training:
    tweet = preprocess(item)
    game_pre_processed_data.append(tweet)

# vectorizing the cleaned training data
X = vectorizer(game_pre_processed_data)



# dividing the training data into training and test set. test set is 20% of the training data
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=0)


# initializing the RandomForestClassifier with parameters of max_feature and n_estimators
text_classifier = RandomForestClassifier(max_features=1844, n_estimators=50)
# training the model with training-data
text_classifier.fit(X_train, y_train)
# making prediction with the trained model.
predictions = text_classifier.predict(X_test)
# testing the model by comparing the predictions with test labels
# print(confusion_matrix(y_test,predictions))
print("Accuracy rate of the model trained with manually labeled data set")
print(accuracy_score(y_test, predictions))


# counting the polarities in the prediction
Positive = 0
Negative = 0
Neutral = 0
for item in predictions:
    if item == 0:
        Neutral += 1
    elif item == -1:
        Negative += 1
    else:
        Positive += 1


# using the counted values to draw result by using matplotlib on the screen
results = [Negative, Neutral, Positive]
plt.style.use('ggplot')
x = ['Negative', 'Neutral', 'Positive']
x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, results, color='green')
plt.xlabel("Sentiment")
plt.ylabel("Sentiment Count")
plt.title("Sentiment Analysis")
plt.xticks(x_pos, x)
plt.show()


# this part includes random forest model trainged with airline tweets data set
print("=========================================================================")
print("Model trained with airline tweets")

# reading the data
data = "archive/Tweets.csv"
data_read = pd.read_csv(data)
# 10. column of the data is tweets
tweets = data_read.iloc[:, 10].values
# 1. column of the data is the sentiment values
sentiments = data_read.iloc[:, 1].values
# data is limited to 3000 tweets to train the model
tweets = tweets[: 3000]
sentiments = sentiments[:3000]

# list to hold the cleaned data
pre_processed_data = []
# process of cleaning each tweet in the training set
for item in tweets:
    tweet = preprocess(item)
    pre_processed_data.append(tweet)
# vectorizing the cleaned training data
X = vectorizer(pre_processed_data)

# sentiments were strings, they were converted to -1 for negatives, 0 for neutrals, 1 for positives
y = []
for item in sentiments:
    if item == "positive":
        y.append(1)
    elif item == "negative":
        y.append(-1)
    else:
        y.append(0)

# initializing the RandomForestClassifier
text_classifier2 = RandomForestClassifier()
# training the model with training-data
text_classifier2.fit(X, y)

# using the model trained with airlines data set, predictions were made in manually labeled test data.
predictions = text_classifier2.predict(X_test)

# testing the model by comparing the predictions with test labels
# print(confusion_matrix(y_test,predictions))
print(accuracy_score(y_test, predictions))

print("===================================================")
# Whole data Prediction


game_data = "archive/tag_englishTweets.csv"
game_read = pd.read_csv(game_data)

game_tweets = game_read.text.to_numpy()

game_pre_processed_data = []
for item in game_tweets:
    tweet = preprocess(item)
    game_pre_processed_data.append(tweet)

X = vectorizer(game_pre_processed_data)

# predictions were made by the model trained with manually labeled training data set
predictions = text_classifier.predict(X)
print("Sentiment analysis made by the model trained with manually labeled training data")

Positive = 0
Negative = 0
Neutral = 0

for item in predictions:

    if item == 0:
        Neutral += 1
    elif item == -1:
        Negative += 1
    else:
        Positive += 1

results = [Negative, Neutral, Positive]

plt.style.use('ggplot')

x = ['Negative', 'Neutral', 'Positive']

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, results, color='green')
plt.xlabel("Sentiment")
plt.ylabel("Sentiment Count")
plt.title("Sentiment Analysis")

plt.xticks(x_pos, x)

plt.show()