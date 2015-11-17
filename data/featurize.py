from collections import Counter
import re

from sklearn.feature_extraction.text import CountVectorizer

from sentiment import Sentiment
from util import count_links

class Featurizer(object):
    def __init__(self):
        self.sentiment_analyzer = Sentiment('AFINN-111.txt')
        self.bow_vectorizer = None
        self.bow_analyzer = None

    def enhance(self, comment):
        """ Input: comment structure directly from dataset
            Output: enhanced comment structure """

        if not self.bow_vectorizer or not self.bow_analyzer:
            print 'Please call transform() instead'
            return

        comment["num_chars"] = len(comment.get("body"))
        comment["num_links"] = count_links(comment.get("body"))

        # To calculate num words, do a simple split on spaces. Bag of words
        # will require more sophisticated tokenization and remove stopwords
        simple_tokens = comment.get("body").split(' ')
        num_words = 0
        avg_word_length = 0
        for token in simple_tokens:
            num_words += 1
            avg_word_length += len(token)
        comment["num_words"] = num_words
        comment["avg_word_length"] = float(avg_word_length) / float(num_words)

        # Use scikit-learn to tokenize and filter stopwords
        comment["tokens"] = self.bow_analyzer(comment.get("body"))
        comment["sentiment"] = \
            self.sentiment_analyzer.analyze(comment.get("tokens"))
        comment["bag_of_words"] = self.bow_vectorizer.transform([comment.get("body")])

        return comment

    def transform(self, comments):

        # http://scikit-learn.org/stable/modules/feature_extraction.html
        self.bow_vectorizer = CountVectorizer(min_df=1)
        self.bow_vectorizer.fit([c.get("body") for c in comments])
        self.bow_analyzer = self.bow_vectorizer.build_analyzer()
