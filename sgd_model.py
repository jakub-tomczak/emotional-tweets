from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from model import ScikitModel
import numpy as np


class SGDModel(ScikitModel):
    def train(self):
        vect = HashingVectorizer(decode_error='ignore',
                                 preprocessor=None,
                                 analyzer='word',
                                 tokenizer=self.tweet_preprocessor)

        mlb = preprocessing.LabelEncoder()
        y_train = mlb.fit_transform(self.y_train)
        y_test = mlb.fit_transform(self.y_test)
        # y_train = mlb.transform(y_train)
        clf = SGDClassifier(loss='log', random_state=1, max_iter=20)
        self.model = clf
        batch_size = 100
        training_samples = self.X_train.shape[0]

        for i in range(batch_size):
            start = i * batch_size
            end = min((i + 1) * batch_size, training_samples)
            if abs(start - end) < 2:
                break
            x = vect.transform(self.X_train[start:end])
            y = y_train[start:end]
            clf.partial_fit(x, y, classes=[0, 1, 2])

        X_test = vect.transform(self.X_test)
        print('Dokładność testu: %.3f' % clf.score(X_test, y_test))

    def test(self, data):
        return np.array(self.model.predict(data))
