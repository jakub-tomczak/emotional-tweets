import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt

categories = {'negative': [1, 0, 0], 'neutral': [0, 1, 0], 'positive': [0, 0, 1]}

# categories ={'negative':-1,'neutral':0,'positive':1}
glove_filename = 'data/glove.6B.100d'


def save_glove():
    import pickle
    from numpy import asarray
    embeddings_dictionary = dict()
    with open(f'{glove_filename}.txt', encoding="utf8") as glove_file:
        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = asarray(records[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions

    with open(f'{glove_filename}.pkl', 'wb') as f:
        pickle.dump(embeddings_dictionary, f, protocol=4)


def load_glove(vocab_size, tokenizer):
    path = f'{glove_filename}.pkl'
    if os.path.exists(path):
        from numpy import zeros
        with open(path, 'rb') as f:
            try:
                data = pickle.load(f)

                embedding_matrix = zeros((vocab_size, 100))
                for i in range(len(tokenizer.subwords)):
                    embedding_vector = data.get(tokenizer.subwords[i])
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector
                # for Tokenizer from keras textPreprocessing
                # for word, index in tokenizer.word_index.items():
                #     embedding_vector = data.get(word)
                #     if embedding_vector is not None:
                #         embedding_matrix[index] = embedding_vector
                print('loaded from pickle')
                return embedding_matrix
            except Exception as e:
                print(f"Failed to load processed data from {path}")
    print('glove file doesn\'t extis')
    exit(-1)


def load_processed_data(data_dir) -> (pd.DataFrame, pd.DataFrame):
    def load_file(filename):
        path = os.path.join(data_dir, f'{filename}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                try:
                    data = pickle.load(f)
                    print('loaded from pickle')
                    return data
                except:
                    print(f"Failed to load processed data from {path}")
                    return None
        else:
            return None

    return load_file('train'), load_file('test')


def load_dataset(data_dir) -> (pd.DataFrame, pd.DataFrame):
    def load_file(filename):
        data = pd.read_csv(os.path.join(data_dir, f'{filename}.csv'), dtype=str)
        data = data.loc[pd.notna(data['Id'])]
        return data

    train = load_file('train')
    train['class'] = train.apply(classes, axis=1)

    test = load_file('test')
    print('loaded from csv files')
    return train, test


def histograms(data):
    print(data.groupby('Category').count())


def classes(row):
    try:
        return categories[row['Category']]
    except:
        return None


if __name__ == '__main__':
    train, test = load_dataset()
    histograms(train)
    print(train.head)
