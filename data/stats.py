import pprint

import featurize
import util

if __name__ == "__main__":
    # compsci = util.load_comments_from_file("compsci.csv")
    hiphop = util.load_comments_from_file("hiphopheads.csv")
    hiphop = sorted(hiphop, key=lambda x: int(x["score"]), reverse=True)
    f = featurize.Featurizer()

    f.transform(hiphop)
    
    print 'Number of comments: {}'.format(len(hiphop))
    print 'Highest score: {}'.format(hiphop[0]["score"])
    print 'Top five comments:'
    for c in hiphop[:5]:
        pprint.pprint(f.enhance(c))

