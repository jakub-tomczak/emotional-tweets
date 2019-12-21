import nltk


def initialize_nltk():
    # nltk.download('opinion_lexicon')
    # nltk.download('sentiwordnet')
    if not (nltk.download('punkt') and nltk.download('stopwords')):
        print("Initialization failed")
        return False
    print('nltk Initialized')
    return True
