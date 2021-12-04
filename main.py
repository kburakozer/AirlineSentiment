from model import Model

def main():
    path = "/home/kburakozer/Documents/SWE/SWE/3. Term/SWE599/AirlineSentiment/archive/Tweets.csv"
    path2 = "/home/kburakozer/Documents/SWE/SWE/3. Term/SWE599/AirlineSentiment/archive/NewsCategorizer.csv"
    path3 = "/home/kburakozer/Documents/SWE/SWE/3. Term/SWE599/AirlineSentiment/archive/IMDB Dataset.csv"
    # airline_model = Model(path, "text", "airline_sentiment") ## 10000 - 25
   
    news_model = Model(path2, "short_description", "category")
    # imdb_model = Model(path3, "review", "sentiment")
    
    result = news_model.LSTM()
    print(result)

if __name__ == "__main__":
    main()