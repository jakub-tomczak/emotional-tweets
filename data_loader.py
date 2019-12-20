import pandas as pd


def load_dataset() -> (pd.DataFrame, pd.DataFrame):
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test
