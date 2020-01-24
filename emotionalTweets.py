from data_loader import load_dataset, load_processed_data
from logistic_regression_model import LogisticRegressionModel
from neural_network_model import NeuralNetworkModel
from nltk_helper import initialize_nltk
from tokenizers import TweetTokenizer
from steemers import DefaultStemmer
from transformers import EmoticonsTransformer
from words_filter import NltkStopwordsFilter, HyperLinkFilter, HashTagFilter, \
    SpecialCharactersFilter
import os


def save_processed_data(train_data, test_data):
    def save_data(data, filename):
        import pickle
        path = os.path.join('data', f'{filename}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=4)

    a = process_tweets(train_data)
    save_data(a, 'train')
    print('saved train processed data')

    # don't remove Not Available lines - they need to be in submission
    b = process_tweets(test_data, remove_not_available_tweets=False)
    save_data(b, 'test')
    print('saved test processed data')


def process_tweet(tweet):
    tokens = TweetTokenizer.tokenize(tweet)
    removed_special_chars = SpecialCharactersFilter.filter(tokens)
    filtered_tweets = NltkStopwordsFilter.filter(removed_special_chars)
    filtered_tweets = HyperLinkFilter.filter(filtered_tweets)
    filtered_tweets = HashTagFilter.filter(filtered_tweets)
    transformed = EmoticonsTransformer.transform(filtered_tweets)
    return DefaultStemmer.stem(transformed)


def process_tweets(data, remove_not_available_tweets=True):
    new_data = data.loc[data.Tweet != "Not Available"] \
        if remove_not_available_tweets \
        else data
    new_data['processed'] = new_data['Tweet'].apply(process_tweet)
    return new_data


def analyse_data(data):
    import matplotlib.pyplot as plt
    data.Category.value_counts().plot(kind='pie', autopct='%1.0f%%',
                                      colors=["red", "yellow", "green"])
    plt.show()


def test_for_submission(model, data, output_dir='data'):
    # classify Not Available as positive class
    # not_available_tweets = data.loc[data.Tweet == "Not Available"]
    # not_available_tweets['results'] = 'positive'
    #
    # # get rows without Not Available
    # data = data.loc[data.Tweet != "Not Available"]

    if model.transformations_reqiured:
        tweets = data.Tweet.apply(process_tweet)
        tested = model.test(tweets)
        data['results'] = tested
    else:
        data['results'] = model.test(data.Tweet)

    # final_data = pd.concat([not_available_tweets, data])
    final_data = data
    final_data.to_csv(
        path_or_buf=os.path.join(output_dir, f'{model.name}_submission.csv'),
        columns=['Id', 'results'],
        header=['Id', 'Category'],
        index=False
    )
    return data


def main():
    # preprocess glove
    # from data_loader import save_glove, load_glove
    # save_glove()
    # glove = load_glove()

    exit(-1)
    model_requires_transformed_data = True
    # if true, then we try to load processed data
    try_loading_processed_data = True and model_requires_transformed_data

    train, test = load_processed_data('data') if try_loading_processed_data \
        else load_dataset('data')

    # save processed data and save it as pickle object
    # train data is processed without Not Available tweets
    # test data is processed with Not Available tweets to keep right submission's shape
    # save_processed_data(train, test)

    if any([train is None, test is None]):
        print('Failed to load dataset')
        exit(-1)

    if not initialize_nltk():
        exit(1)

    model = NeuralNetworkModel(train,
                               process_tweet,
                               transformations_required=model_requires_transformed_data)

    if not model.try_loading_model():
        print(f'model not saved before {model.name}')
        model.train()
        model.save_model()
    else:
        print(f'loaded model: {model.name}')
    test_for_submission(model, test)


if __name__ == "__main__":
    main()
