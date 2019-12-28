import pandas as pd
import matplotlib.pyplot as plt


categories ={'negative':[1,0,0],'neutral':[0,1,0],'positive':[0,0,1]}


def load_dataset() -> (pd.DataFrame, pd.DataFrame):
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    train['class'] = train.apply(classes,axis=1)
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