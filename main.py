from model import Model

def main():
    model_class = Model()
    # train_sentences, val_sentences, train_labels, val_labels = model_class.data_set()
    # result = model_class.bayes_model()
    # print(result)

    result = model_class.LSTM()
    print(result)


if __name__ == "__main__":
    main()