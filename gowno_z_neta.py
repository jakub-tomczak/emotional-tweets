#important libraries
import pandas as pd
import numpy as np
import nltk
import re
#importing stopwords is optional, in this case it decreased accuracy
#from nltk.corpus import stopwords
import itertools
import time


import os
from nltk.stem.wordnet import WordNetLemmatizer 
from data_loader import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import train_test_split
def cleaning(text):
    txt = str(text)
    txt = re.sub(r"http\S+", "", txt)
    if len(txt) == 0:
        return 'no text'
    else:
        txt = txt.split()
        index = 0
        for j in range(len(txt)):
            if txt[j][0] == '@':
                index = j
        txt = np.delete(txt, index)
        if len(txt) == 0:
            return 'no text'
        else:
            words = txt[0]
            for k in range(len(txt)-1):
                words+= " " + txt[k+1]
            txt = words
            txt = re.sub(r'[^\w]', ' ', txt)
            if len(txt) == 0:
                return 'no text'
            else:
                txt = ''.join(''.join(s)[:2] for _, s in itertools.groupby(txt))
                txt = txt.replace("'", "")
                txt = nltk.tokenize.word_tokenize(txt)
                #data.content[i] = [w for w in data.content[i] if not w in stopset]
                for j in range(len(txt)):
                    txt[j] = lem.lemmatize(txt[j], "v")
                if len(txt) == 0:
                    return 'no text'
                else:
                    return txt






if __name__ == '__main__':
    lem = WordNetLemmatizer()
    start_time = time.time()
    train, test = load_dataset()
    train = train[~(pd.isna(train.Tweet))]

    train['Tweet'] = train['Tweet'].map(lambda x: cleaning(x))
    train = train.reset_index(drop=True)
    for i in range(len(train)):
        words = train.Tweet[i][0]
        for j in range(len(train.Tweet[i])-1):
            words+= ' ' + train.Tweet[i][j+1]
        train.Tweet[i] = words


    x_train, x_test, y_train, y_test = train_test_split(train.Tweet, train['Category'], test_size=0.25, random_state=0)
    x_train = x_train.reset_index(drop = True)
    x_test = x_test.reset_index(drop = True)

    y_train = y_train.reset_index(drop = True)
    y_test = y_test.reset_index(drop = True)
    
    vectorizer = TfidfVectorizer(min_df=3, max_df=0.9)
    train_vectors = vectorizer.fit_transform(x_train)
    test_vectors = vectorizer.transform(x_test)
    model = svm.SVC(kernel='linear') 
    model.fit(train_vectors, y_train) 
    predicted_sentiment = model.predict(test_vectors)

    print(classification_report(y_test, predicted_sentiment))

    predicted_sentiments = []
    for s in range(len(predicted_sentiment)):
        predicted_sentiments.append(predicted_sentiment[s])
        
    prediction_df = pd.DataFrame({'Content':x_test, 'Emotion_predicted':predicted_sentiment, 'Emotion_actual': y_test})
    prediction_df.to_csv('emotion_recognizer_svm.csv', index = False)

    elapsed_time = time.time() - start_time
    print ("processing time:", elapsed_time, "seconds")

