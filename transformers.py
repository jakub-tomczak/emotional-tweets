class Transformer:
    @staticmethod
    def transform(tokens):
        pass


class EmoticonsTransformer(Transformer):
    @staticmethod
    def transform(tokens):
        '''
        Removes noses from emoticons.
        :param tokens:
        :return:
        '''
        return [token.replace('-', '') for token in tokens]
