from pyspark import SparkContext

from util import load_comments_from_file
from spark_featurize import SparkFeaturizer

if __name__ == '__main__':

    sc = SparkContext("local", "Prediction",
                      pyFiles = ['util.py', 'spark_featurize.py'])

    # parser = argparse.ArgumentParser(description='Reddit Comment Prediction')
    # parser.add_argument('-i', '--input_file', type=str, 
    #     help="""The CSV input data file that contains the raw comment data""")
    # args = parser.parse_args()

    print 'Loading and processing comments...'
    data = sc.textFile("mllib
