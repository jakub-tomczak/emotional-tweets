from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing


def fit(X_train, y_train, X_test, y_test, tweet_processor):
    vect = HashingVectorizer(decode_error='ignore',
                             preprocessor=None,
                             analyzer='word',
                             tokenizer=tweet_processor)

    mlb = preprocessing.LabelEncoder()
    y_train = mlb.fit_transform(y_train)
    y_test = mlb.fit_transform(y_test)
    # y_train = mlb.transform(y_train)
    clf = SGDClassifier(loss='log', random_state=1, max_iter=20)

    batch_size = 100
    training_samples = X_train.shape[0]

    for i in range(batch_size):
        start = i * batch_size
        end = min((i + 1) * batch_size, training_samples)
        if abs(start - end) < 2:
            break
        x = vect.transform(X_train[start:end])
        y = y_train[start:end]
        clf.partial_fit(x, y, classes=[0, 1, 2])

    X_test = vect.transform(X_test)
    print('Dokładność testu: %.3f' % clf.score(X_test, y_test))
