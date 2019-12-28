from keras.models import Sequential
from keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from data_loader import *
from nltk.stem.lancaster import LancasterStemmer
import nltk
from emotionalTweets import process_tweets
from nltk.stem.lancaster import LancasterStemmer


categories ={'negative':[1,0,0],'neutral':[0,1,0],'positive':[0,0,1]}

def get_model(input_dim):
    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train(model, X_train, y_train,X_test,y_test):
    history = model.fit(X_train, y_train, epochs=100, verbose=False, validation_data=(X_test, y_test),batch_size=10)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))


