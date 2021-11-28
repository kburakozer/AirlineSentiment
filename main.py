from model import Model

def main():
    path = "/home/kburakozer/Documents/SWE/SWE/3. Term/SWE599/AirlineSentiment/archive/Tweets.csv"
    airline_model = Model(path, "text", "airline_sentiment")
    news_model = Model(path, "short_description", "category")
    # train_sentences, val_sentences, train_labels, val_labels = news_model.data_set()
    result = airline_model.bayes_model()
    print(result)

    # result = news_model.LSTM()
    # print(result)

if __name__ == "__main__":
    main()