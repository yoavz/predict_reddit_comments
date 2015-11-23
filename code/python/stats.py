import pprint

import featurize
import util

if __name__ == "__main__":
    # compsci = util.load_comments_from_file("data/compsci.csv")
    hiphop = util.load_comments_from_file("data/hiphopheads.csv")
    hiphop = sorted(hiphop, key=lambda x: int(x["score"]), reverse=True)
    
    print 'Number of comments: {}'.format(len(hiphop))
    print 'Highest score: {}'.format(hiphop[0]["score"])
    print 'Top five comments:'

