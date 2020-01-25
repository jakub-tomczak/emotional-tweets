import pickle
import os


class Model:
    def __init__(self, train_data, tweet_processor_function, percentage_of_train_data=.7,
                 transformations_required=True):
        '''
        Base class for models.
        :param train_data:
        :param tweet_processor_function:
        :param percentage_of_train_data:
        :param transformations_required: Is transformation required before putting data into fit or predict method.
        '''
        self.model = None
        self.train_data = train_data
        self.number_of_samples = self.train_data.shape[0]
        self.train_num = 0
        self.test_num = 0
        self.transformations_reqiured = transformations_required
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.split_into_train_and_test(percentage_of_train_data)
        self.tweet_preprocessor = tweet_processor_function

    def split_into_train_and_test(self, percentage_of_train_data):
        '''
        Splits self.train_data into self.X_train, self.y_train and self.X_test, self.y_test
        with respect to the percentage_of_train_data arg.
        :param percentage_of_train_data:
        :return: None
        '''
        self.train_num = int(self.number_of_samples * percentage_of_train_data)
        self.test_num = self.number_of_samples - self.train_num
        self.X_train, self.y_train = self.train_data[:self.train_num].processed, self.train_data[:self.train_num].Category
        self.X_test, self.y_test = self.train_data[self.train_num:].processed, self.train_data[self.train_num:].Category

    def train(self):
        '''
        :return: nothing
        '''
        pass

    def evaluate_model(self):
        pass

    def test(self, data):
        pass

    def save_model(self, name='', save_dir='models'):
        pass

    def try_loading_model(self, name='', load_dir='models'):
        pass

    def model_name(self, name=''):
        return self.name if name == '' else name

    @property
    def name(self):
        return Model.get_model_name(self)

    @staticmethod
    def get_model_name(model):
        return type(model).__name__


class ScikitModel(Model):
    def save_model(self, name='', save_dir='models'):
        path = os.path.join(save_dir, self.model_name(name))
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(save_dir)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f, protocol=4)

    def try_loading_model(self, name='', load_dir='models'):
        path = os.path.join(load_dir, self.model_name(name))
        if os.path.exists(path):
            with open(path, 'rb') as f:
                try:
                    self.model = pickle.load(f)
                except:
                    print("Failed to read model")
                    return False
            return True
        else:
            return False
