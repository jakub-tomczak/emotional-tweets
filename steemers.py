import nltk


class Stemmer():
    def __init__(self):
        self.stemmer = None

    @staticmethod
    def stem(tokens):
        pass


class DefaultStemmer(Stemmer):
    @staticmethod
    def stem(tokens):
        stemmer = nltk.PorterStemmer()
        return [stemmer.stem(token) for token in tokens]
