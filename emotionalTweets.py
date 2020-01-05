from data_loader import load_dataset
from logistic_regression_model import LogisticRegressionModel
from nltk_helper import initialize_nltk
from tokenizers import TweetTokenizer
from steemers import DefaultStemmer
from transformers import EmoticonsTransformer
from words_filter import NltkStopwordsFilter, HyperLinkFilter, HashTagFilter, \
    SpecialCharactersFilter
import os


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


def analyse_data(data):
    import matplotlib.pyplot as plt
    data.Category.value_counts().plot(kind='pie', autopct='%1.0f%%',
                                      colors=["red", "yellow", "green"])
    plt.show()


def test_for_submission(model, data, output_dir='data'):
    processed_tweets = data.Tweet.apply(process_tweet)
    data['results'] = model.test(processed_tweets)
    data.to_csv(
        path_or_buf=os.path.join(output_dir, 'submission.csv'),
        columns=['Id', 'results'],
        header=['Id,Category']
    )
    return data


def main():
    train, test = load_dataset()

    train = train
    if not initialize_nltk():
        exit(1)

    model = LogisticRegressionModel(train, process_tweet)
    model.train()
    # if not model.try_loading_model():
    #     print('model not saved before')
    #     model.train()
    #     model.save_model()
    # else:
    #     print('loaded model')
    # test_for_submission(model, test)


if __name__ == "__main__":
    main()
