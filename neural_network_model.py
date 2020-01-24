import os

from model import Model
from keras.models import Sequential
from keras import layers
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

import pickle


class NeuralNetworkModel(Model):
    categories_encode = {
        'negative': [1, 0, 0],
        'neutral': [0, 1, 0],
        'positive': [0, 0, 1]
    }

    categories_decode = {
        0: 'negative',
        1: 'neutral',
        2: 'positive'
    }

    def __init__(self, train_data, tweet_processor_function, percentage_of_train_data=.9,
                 transformations_required=True):
        super().__init__(train_data, tweet_processor_function,
                         percentage_of_train_data=percentage_of_train_data,
                         transformations_required=True)

        self.max_length_encoding = 200
        self.subname = 'lstm'
        p = os.path.join('data', 'data_tokenizer.pkl')
        text_encoder = None
        if os.path.exists(p):
            with open(p, 'rb') as f:
                print(f'read {p}')
                text_encoder = pickle.load(f)
        else:
            text_encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (' '.join(x) for x in train_data.processed), target_vocab_size=2 ** 13)
            with open(p, 'wb') as f:
                print(f'saved {p}')
                pickle.dump(text_encoder, f, protocol=4)
        self.data_tokenizer = text_encoder

        def load_or_encode(data, name, encode_method):
            '''
            Loading / saving X_train, X_test, y_train, y_test
            to avoid encoding those data again and again.
            If self.percentage_of_train_data has been changed
            the data must not be loaded since saved data have different shape
            that required.
            :param data:
            :param name:
            :param encode_method:
            :return:
            '''
            p = os.path.join('data', f'{name}.pkl')
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    print(f'read {p}')
                    loaded = pickle.load(f)
                    return loaded
            encoded_data = encode_method(data)
            with open(p, 'wb') as f:
                print(f'saved {p}')
                pickle.dump(encoded_data, f, protocol=4)
            return encoded_data

        self.X_train = load_or_encode(self.X_train, 'nn_x_train', self.encode_x_data)
        self.X_test = load_or_encode(self.X_test, 'nn_x_test', self.encode_x_data)
        self.y_train = load_or_encode(self.y_train, 'nn_y_train', NeuralNetworkModel.encode_y_data)
        self.y_test = load_or_encode(self.y_test, 'nn_y_test', NeuralNetworkModel.encode_y_data)

        self.model = NeuralNetworkModel.get_model(self.X_train.shape[-1], self.data_tokenizer.vocab_size)

    def encode_x_data(self, data):
        return tf.keras.preprocessing.sequence.pad_sequences([self.data_tokenizer.encode(' '.join(x)) for x in data],
                                                             padding='post',
                                                             maxlen=self.max_length_encoding)

    def train(self):
        save_callback = tf.keras.callbacks.ModelCheckpoint(
            self.model_filename,
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            period=1)
        self.model.fit(self.X_train, self.y_train,
                       epochs=40,
                       validation_data=(self.X_test, self.y_test),
                       batch_size=100,
                       callbacks=[save_callback])
        loss, accuracy = self.model.evaluate(self.X_train, self.y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

    def test(self, data):
        data = self.encode_x_data(data)
        result = self.model.predict(data)
        result = tf.math.argmax(result, axis=1).numpy()
        return [NeuralNetworkModel.categories_decode[x] for x in result]

    @property
    def model_filename(self):
        return os.path.join('models', f'{self.name}_{self.subname}.h5')

    def save_model(self, name='', save_dir='models'):
        path = self.model_filename
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(save_dir)
        self.model.save_weights(path)

    def try_loading_model(self, name='', load_dir='models'):
        path = self.model_filename
        if os.path.exists(path):
            with open(path, 'rb') as f:
                try:
                    self.model.load_weights(path)
                    print('loaded model succesfully', path)
                except Exception as e:
                    print("Failed to read model")
                    return False
            return True
        else:
            return False

    @staticmethod
    def encode_y_data(data):
        return np.array([NeuralNetworkModel.categories_encode[item] for item in data])

    @staticmethod
    def get_model(input_size, vocabulary_size):
        print(f'creating model with vocabulary size of {vocabulary_size} and input size {input_size}')
        # model = Sequential([
        #     layers.Embedding(vocabulary_size, 16, input_shape=(input_size,)),
        #     # layers.Flatten(),
        #     # layers.Dense(128, activation='relu'),
        #     layers.Dense(3, activation='softmax')
        # ])

        model = Sequential([
            layers.Embedding(vocabulary_size, 64),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dense(64, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss='mse',
                      metrics=['accuracy'])
        model.summary()

        return model

    def set_data(X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.y_test = y_test
        self.X_test = X_test
