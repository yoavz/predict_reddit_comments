import argparse

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import Tokenizer

def filter_comments(df):
    return df.filter(df['author'] != '[deleted]') \
             .filter(df['body'] != '[deleted]') \
             .filter(df['body'] != '[removed]')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Reddit Comment Prediction')
    parser.add_argument('-i', '--input_file', type=str, 
        help="""The CSV input data file that contains the raw comment data""")
    args = parser.parse_args()

    sc = SparkContext("local", "Prediction")
    sqlContext = SQLContext(sc)
    df = sqlContext.read.json(args.input_file)
    print 'Loaded input file {} with {} total comments'.format(args.input_file, df.count())

    filtered = filter_comments(df)
    print '{} comments after filtering'.format(filtered.count())

    tokenizer = Tokenizer(inputCol="body", outputCol="words")
    wordsDataFrame = tokenizer.transform(filtered)
    wordsDataFrame.select("body", "words").show() 
