from featurize import Featurizer

class SparkFeaturizer(Featurizer):
    
    def transform(self, comments):
        """ Returns a Nx(D+1) matrix of features. The first D columns
        correspond to features, where the final column corresponds to the
        scores of each comment"""
        pass
