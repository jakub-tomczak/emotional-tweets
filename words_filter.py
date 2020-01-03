import re
from tokenizers import RE_PUNCTUATIONS, RE_HTTP, RE_HASHTAG, RE_WWW, RE_SPECIAL


class WordsFilter:
    @staticmethod
    def filter(tokens):
        '''
        Method that removes some tokens from tokens list.
        :param tokens: list of tokens ['A', 'short', 'list']
        :return: flitered tokens
        '''
        pass

    @staticmethod
    def remove_list_of_words(structure, list_of_items):
        for item in list_of_items:
            if type(structure) is list:
                structure.remove(item)
            elif type(structure) is dict:
                del structure[item]
        return structure

    @staticmethod
    def remove_words_using_list(tokens_collection, words_to_remove_list):
        keys_to_remove = [k for k in tokens_collection if k in words_to_remove_list]
        return WordsFilter.remove_list_of_words(tokens_collection, keys_to_remove)

    @staticmethod
    def remove_words_using_regex(tokens_collection, regex):
        keys_to_remove = [k for k in tokens_collection if re.match(regex, k)]
        return WordsFilter.remove_list_of_words(tokens_collection, keys_to_remove)


class DefaultStopwordsFilter(WordsFilter):
    stopwords = ["a", "about", "after", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because",
                 "been",
                 "before", "being", "between", "both", "by", "could", "did", "do", "does", "doing", "during",
                 "each",
                 "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
                 "herself", "him",
                 "himself", "his", "how", "i", "in", "into", "is", "it", "its", "itself", "let", "me", "more",
                 "most", "my",
                 "myself", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
                 "own", "sha",
                 "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
                 "themselves",
                 "then", "there", "there's", "these", "they", "this", "those", "through", "to", "until", "up",
                 "very",
                 "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "with",
                 "would", "you",
                 "your", "yours", "yourself", "yourselves",
                 "n't", "'s", "'ll", "'re", "'d", "'m", "'ve",
                 "above", "again", "against", "below", "but", "cannot", "down", "few", "if", "no", "nor",
                 "not", "off",
                 "out", "over", "same", "too", "under", "why"]

    @staticmethod
    def filter(tokens):
        return WordsFilter.remove_words_using_list(tokens, DefaultStopwordsFilter.stopwords)


class NltkStopwordsFilter(WordsFilter):
    @staticmethod
    def filter(tokens):
        from nltk.corpus import stopwords
        return WordsFilter.remove_words_using_list(tokens, stopwords.words('english'))


class PunctuationsFilter(WordsFilter):
    @staticmethod
    def filter(tokens):
        return WordsFilter.remove_words_using_regex(tokens, RE_PUNCTUATIONS)


class SpecialCharactersFilter(WordsFilter):
    @staticmethod
    def filter(tokens):
        return WordsFilter.remove_words_using_regex(tokens, RE_SPECIAL)


class HyperLinkFilter(WordsFilter):
    @staticmethod
    def filter(tokens):
        filtered_http = WordsFilter.remove_words_using_regex(tokens, RE_WWW)
        return WordsFilter.remove_words_using_regex(filtered_http, RE_HTTP)


class HashTagFilter(WordsFilter):
    @staticmethod
    def filter(tokens):
        return WordsFilter.remove_words_using_regex(tokens, RE_HASHTAG)
