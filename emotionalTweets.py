from data_loader import load_dataset
from nltk_helper import initialize_nltk
from tokenizers import TweetTokenizer, NltkTokenizer
from steemers import DefaultStemmer
from words_filter import NltkStopwordsFilter, PunctuationsFilter, HyperLinkFilter, HashTagFilter


def process_tweets(data):
    new_data = data.loc[data.Tweet != "Not Available"]
    processed_tweets = []
    for row in new_data.iterrows():
        row = row[1]
        tweet = row[2]
        tokens = TweetTokenizer.tokenize(tweet)
        removed_punctuations = PunctuationsFilter.filter(tokens)
        filtered_tweets = NltkStopwordsFilter.filter(removed_punctuations)
        filtered_tweets = HyperLinkFilter.filter(filtered_tweets)
        filtered_tweets = HashTagFilter.filter(filtered_tweets)
        stemmed = DefaultStemmer.stem(filtered_tweets)
        processed_tweets.append(stemmed)
    return new_data.Category, processed_tweets


def main():
    train, test = load_dataset()

    train = train
    if not initialize_nltk():
        exit(1)

    processed_tweets = process_tweets(train)


if __name__ == "__main__":
    main()
