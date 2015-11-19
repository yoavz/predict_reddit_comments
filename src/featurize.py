from collections import Counter
import re

from sklearn.feature_extraction.text import CountVectorizer
import numpy

from sentiment import Sentiment
from util import count_links

class Featurizer(object):
    def __init__(self):
        self.sentiment_analyzer = Sentiment('data/AFINN-111.txt')
        self.bow_vectorizer = None
        self.bow_analyzer = None

    def bag_of_words(self, body):
        return self.bow_vectorizer.transform([body]).toarray()

    def text_features(self, comment):
        num_chars = len(comment.get("body"))
        num_links = count_links(comment.get("body"))

        simple_tokens = comment.get("body").split(' ')
        num_words = 0
        avg_word_length = 0
        for token in simple_tokens:
            num_words += 1
            avg_word_length += len(token)
        avg_word_length = float(avg_word_length) / float(num_words)

        sentiment = self.sentiment_analyzer.analyze(
            self.bow_analyzer(comment.get("body")))

        score = comment.get("score")

        return [num_chars, num_links, num_words, num_words, 
                avg_word_length, sentiment]

    def transform_comment(self, comment):
        return numpy.hstack((
            numpy.array([self.text_features(comment)], 
                        dtype='float_'),
            self.bag_of_words(comment.get("body"))))

    def score_comment(self, comment):
        return comment.get("score")

    def transform(self, comments):
        """ Returns a Nx(D+1) numpy matrix of features. The first D columns
        correspond to features, where the final column corresponds to the
        scores of each comment"""

        # if it's a single instance, return an array
        if isinstance(comments, dict):
            return transform_comment(comments)

        # http://scikit-learn.org/stable/modules/feature_extraction.html
        self.bow_vectorizer = CountVectorizer(min_df=1)
        self.bow_vectorizer.fit([c.get("body") for c in comments])
        self.bow_analyzer = self.bow_vectorizer.build_analyzer()

        def features_and_label(comment):
            return numpy.hstack((
                self.transform_comment(comment),
                numpy.array([[self.score_comment(comment)]], 
                            dtype='float_')))

        return numpy.vstack([features_and_label(c) 
                             for c in comments])


if __name__ == "__main__":
    import argparse
    pass
