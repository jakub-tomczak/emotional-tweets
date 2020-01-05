from model import Model
from keras.models import Sequential
from keras import layers


class NeuralNetworkModel(Model):
    categories = {'negative': [1, 0, 0], 'neutral': [0, 1, 0], 'positive': [0, 0, 1]}

    def train(self):
        self.model = NeuralNetworkModel.get_model(3)
        self.model.fit(self.X_train, self.y_train, epochs=100, verbose=False,
                            validation_data=(self.X_test, self.y_test),
                            batch_size=10)
        loss, accuracy = self.model.evaluate(self.X_train, self.y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

    def test(self, data):
        return self.model.predict(data)

    @staticmethod
    def get_model(input_dim):
        model = Sequential()
        model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
