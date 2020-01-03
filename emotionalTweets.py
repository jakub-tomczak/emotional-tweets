from data_loader import load_dataset
from nltk_helper import initialize_nltk
from tokenizers import TweetTokenizer
from steemers import DefaultStemmer
from transformers import EmoticonsTransformer
from words_filter import NltkStopwordsFilter, HyperLinkFilter, HashTagFilter, \
    SpecialCharactersFilter


def process_tweet(tweet):
    tokens = TweetTokenizer.tokenize(tweet)
    removed_special_chars = SpecialCharactersFilter.filter(tokens)
    filtered_tweets = NltkStopwordsFilter.filter(removed_special_chars)
    filtered_tweets = HyperLinkFilter.filter(filtered_tweets)
    filtered_tweets = HashTagFilter.filter(filtered_tweets)
    transformed = EmoticonsTransformer.transform(filtered_tweets)
    return DefaultStemmer.stem(transformed)


def process_tweets(data):
    new_data = data.loc[data.Tweet != "Not Available"]
    new_data['processed'] = new_data['Tweet'].apply(process_tweet)
    return new_data


def main():
    train, test = load_dataset()

    train = train
    if not initialize_nltk():
        exit(1)

    #processed_tweets = process_tweets(train[10:20])
    import logistic_regression_model
    X_train, y_train = train[1:3000].Tweet, train[1:3000].Category
    X_test, y_test = train[3001:].Tweet, train[3001:].Category
    logistic_regression_model.fit(X_train, y_train, X_test, y_test, process_tweet)



if __name__ == "__main__":
    main()
