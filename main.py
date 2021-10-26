from model import Model

def main():
    path = "/home/kburakozer/Documents/SWE/SWE/3. Term/SWE599/Airline/archive/Tweets.csv"
    model_class = Model(path)
    # train_sentences, val_sentences, train_labels, val_labels = model_class.data_set()
    # result = model_class.bayes_model()
    # print(result)

    result = model_class.LSTM()
    print(result)


if __name__ == "__main__":
    main()