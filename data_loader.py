import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt

categories = {'negative': [1, 0, 0], 'neutral': [0, 1, 0], 'positive': [0, 0, 1]}

# categories ={'negative':-1,'neutral':0,'positive':1}

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
