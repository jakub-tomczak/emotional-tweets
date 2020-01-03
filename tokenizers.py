import nltk
import re

RE_SPACES = re.compile("\s+")
RE_HASHTAG = re.compile("[@#][_a-zA-Z0-9]+")
RE_EMOTICONS = re.compile("(:-?\))|(:p)|(:d+)|(:-?\()|(:/)|(;-?\))|(<3)|(=\))|(\)-?:)|(:'\()|(8\))|(8-\))")
RE_HTTP = re.compile("http(s)?://[/\.a-z0-9]+")
RE_WWW = re.compile("www\.[a-z0-9\.]+")
RE_PUNCTUATIONS = re.compile("['',.?!\-\+=\(\)\{\}\[\]%$&*~``]")
RE_SPECIAL = re.compile(":[''\-\+=\(\)\{\}\[\]%$&*~``]")


class Tokenizer():
    @staticmethod
    def tokenize(text):
        pass


class SimpleTokenizer(Tokenizer):
    '''
    Splits by spaces.
    '''

    @staticmethod
    def tokenize(text):
        return re.split(RE_SPACES, text)


class NltkTokenizer(Tokenizer):
    '''
    Uses default nltk tokenizer (punkt).
    '''

    @staticmethod
    def tokenize(text):
        return nltk.word_tokenize(text)


class TweetTokenizer(Tokenizer):
    '''
    Tokenizer that splits emoticons, links hashtags and nicknames.
    '''

    @staticmethod
    def tokenize(text):
        tokens = SimpleTokenizer.tokenize(text)
        i = 0
        while i < len(tokens):
            token = tokens[i]
            match = re.search(RE_HASHTAG, token) or \
                    re.search(RE_HTTP, token) or \
                    re.search(RE_EMOTICONS, token)
            if match is not None:
                # extract emoticon, link or hashtag/nickname
                del tokens[i]
                span = match.span()
                # split by match and remove empty tokens
                new_tokens = [tok for tok in (token[:span[0]], token[span[0]:]) if tok != '']
                tokens[i:i] = new_tokens
                # increment i, cause we splitted token into two words
                i += len(new_tokens) - 1
            else:
                del tokens[i]
                t = NltkTokenizer.tokenize(token)
                tokens[i:i] = t

            i += 1

        return tokens


class BeforeTokenizationNormalizer():
    @staticmethod
    def normalize(text):
        text = text.strip().lower()
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&amp;', '&')
        text = text.replace('&pound;', u'£')
        text = text.replace('&euro;', u'€')
        text = text.replace('&copy;', u'©')
        text = text.replace('&reg;', u'®')
        return text
