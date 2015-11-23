import argparse
import math

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import util
from featurize import Featurizer
f = Featurizer()

def load_data(input_file_name):
    return f.transform(util.load_comments_from_file(input_file_name))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Reddit Comment Prediction')
    parser.add_argument('-i', '--input_file', type=str, required=True,
        help="""The CSV input data file that contains the raw comment data""")

    args = parser.parse_args()
    print 'Loading and processing comments...'
    data = load_data(args.input_file)
    features = data[:, :-1]
    scores = data[:, -1]
    N = features.shape[0]
    split = int(math.floor(0.6*N))
    print 'Loaded and processed {} comments'.format(N)

    train_features = features[:split, :]
    train_scores = scores[:split]
    test_features = features[split:, :]
    test_scores = scores[split:]

    print 'Training model...'
    clf = linear_model.LinearRegression()
    clf.fit(train_features, train_scores)

    print 'Mean Squared Error: '
    error = mean_squared_error(test_scores, 
                               clf.predict(test_features))
    print error

    # DEBUG
    raw = util.load_comments_from_file(args.input_file)
    for c in raw[-5:]:
        util.pretty_print_comment(c)
        prediction = clf.predict(f.transform_comment(c))
        print 'Real score: {}, Predicted: {}'.format(c.get("score"),
                                                     prediction)
