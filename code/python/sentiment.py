from collections import Counter

class Sentiment(object):
    def _load_tokens(self, filename):
        """ Given a filename of tab seperated word \t sentiment values, 
            save the info into self.tokens """
        self.tokens = dict()
        with open(filename, 'r') as f:
            for line in f.readlines():
                raw = line.strip().split('\t')
                self.tokens[raw[0]] = int(raw[1])

    def __init__(self, filename):
        self._load_tokens(filename)

    def analyze(self, tokens):
        """ Given a bag of words, returns a sentiment score ranging from -5 to
        5. Input tokens may be a list or dictionary of tokens to counts
        """
        score = 0
        recognized = 0
        
        if isinstance(tokens, list):
            tokens = Counter(tokens)

        for token, count in tokens.iteritems():
            if self.tokens.get(token):
                recognized += count
                score += count * self.tokens[token]
        
        if recognized > 0:
            return float(score) / float(recognized)
        else:
            return 0
