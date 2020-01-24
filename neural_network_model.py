import os

from model import Model
from keras.models import Sequential
from keras import layers
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

import pickle
from data_loader import load_glove

from keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


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
        self.subname = 'mse_glove_model2'
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

        self.glove = load_glove(self.data_tokenizer.vocab_size, self.data_tokenizer)

        self.model = NeuralNetworkModel.get_model(
            self.X_train.shape[-1],
            self.data_tokenizer.vocab_size,
            embedding_matrix=self.glove
        )

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
        history = self.model.fit(self.X_train, self.y_train,
                                 epochs=20,
                                 validation_data=(self.X_test, self.y_test),
                                 batch_size=100,
                                 callbacks=[save_callback])

        self.plot_result(history)
        loss, accuracy, f1 = self.model.evaluate(self.X_train, self.y_train, verbose=False)
        print("Training Accuracy: {:.4f}, f1: {:.4f}".format(accuracy, f1))
        loss, accuracy, f1 = self.model.evaluate(self.X_test, self.y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}, f1: {:.4f}".format(accuracy, f1))

    def test(self, data):
        loss, accuracy, f1 = self.model.evaluate(self.X_test, self.y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}, f1: {:.4f}".format(accuracy, f1))

        data = self.encode_x_data(data)
        result = self.model.predict(data)
        result_val = tf.math.argmax(result, axis=1).numpy()
        return [NeuralNetworkModel.categories_decode[x] for x in result_val]

    def plot_result(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])

        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        plt.savefig(f'plots/{self.subname}_accuracy.png')
        plt.clf()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        plt.savefig(f'plots/{self.subname}_loss.png')
        plt.clf()

        plt.plot(history.history['f1_m'])
        plt.plot(history.history['val_f1_m'])

        plt.title('f1')
        plt.ylabel('f1')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        plt.savefig(f'plots/{self.subname}_f1.png')
        plt.clf()

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
    def get_model(input_size, vocabulary_size, embedding_matrix=None):
        print(f'creating model with vocabulary size of {vocabulary_size} and input size {input_size}')
        # model = Sequential([
        #     layers.Embedding(vocabulary_size, 16, input_shape=(input_size,)),
        #     # layers.Flatten(),
        #     # layers.Dense(128, activation='relu'),
        #     layers.Dense(3, activation='softmax')
        # ])

        # LSTM, accuracy 54% validation
        # model = Sequential([
        #     layers.Embedding(vocabulary_size, 64),
        #     layers.Bidirectional(layers.LSTM(64)),
        #     layers.Dense(64, activation='relu'),
        #     layers.Dense(3, activation='softmax')
        # ])

        # on test: acc: ~53%, f1 ~28%
        model = Sequential()
        model.add(layers.Embedding(vocabulary_size, 100, input_length=200, weights=[embedding_matrix])) #, weights=[embedding_matrix] #for glove
        model.add(layers.Conv1D(128, 5, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(3, activation='softmax'))


        # model = Sequential()
        # model.add(layers.Embedding(vocabulary_size, 100, input_length=200, weights=[embedding_matrix]))  # , weights=[embedding_matrix] #for glove
        # model.add(layers.LSTM(100))
        # model.add(layers.Dense(3, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss='mse',
                      metrics=['accuracy', f1_m])
        model.summary()

        return model
