from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import numpy as np

from model import ScikitModel


class LogisticRegressionModel(ScikitModel):
    def train(self):
        tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
        param_grid = [{'vect__ngram_range': [(1, 1)],
                       'vect__stop_words': [None],
                       'vect__tokenizer': [self.tweet_preprocessor],
                       'clf__penalty': ['l2'],
                       'clf__C': [1.0, 10.0, 100.0]},
                      {'vect__ngram_range': [(1, 1)],
                       'vect__stop_words': [None],
                       'vect__tokenizer': [self.tweet_preprocessor],
                       'vect__use_idf': [False],
                       'vect__norm': [None],
                       'clf__penalty': ['l2'],
                       'clf__C': [1.0, 10.0, 100.0]},
                      ]

        lr_tfidf = Pipeline([('vect', tfidf),
                             ('clf', LogisticRegression(random_state=0, max_iter=500))])

        gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                                   scoring='accuracy',
                                   cv=5, verbose=1,
                                   n_jobs=4)
        gs_lr_tfidf.fit(self.X_train, self.y_train)
        print('Zestaw najlepszych parametrów: %s ' % gs_lr_tfidf.best_params_)
        print('Dokładność sprawdzianu krzyżowego: %.3f' % gs_lr_tfidf.best_score_)
        clf = gs_lr_tfidf.best_estimator_
        self.model = clf
        print('Dokładność testu: %.3f' % clf.score(self.X_test, self.y_test))

    def test(self, data):
        return np.array(self.model.predict(data))
