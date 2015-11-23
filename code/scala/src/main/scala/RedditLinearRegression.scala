import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext._
import org.apache.spark.sql.{Row, Column, SQLContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel,
                                    Tokenizer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression

import org.apache.spark.ml.feature.CommentTransformer

object RedditLinearRegression {
  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("Linear Regression")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc);

    val df = sqlContext.read.json(args(0));
    println(s"Loaded input file ${args(0)}");
    println(s"${df.count()} total comments");

    // Filtering logic for removed and deleted comments
    val filtered = df.filter(df("author") !== "[deleted]")
                     .filter(df("body") !== "[removed]")
                     .filter(df("body") !== "[deleted]");
    println(s"${filtered.count()} total comments after filtering");

    val train_to_test = 0.6;
    val splits = filtered.randomSplit(Array(train_to_test, 1-train_to_test));
    val train = splits(0);
    val test = splits(1);
    println(s"Split into ${train.count()} training and ${test.count()} test comments");

    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCol("body")
      .setOutputCol("words");

    val cvModel: CountVectorizer = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("words_features");

    val processor: CommentTransformer = new CommentTransformer()
      .setWordsCol("words")
      .setBodyCol("body")
      .setScoreCol("score_double")
      .setWordsCountCol("words_count")
      .setCharsCountCol("chars_count")
      .setAvgWordLengthCol("avg_word_length")
      .setLinkCountCol("link_count")

    val assembler = new VectorAssembler()
      .setInputCols(Array("words_count", "chars_count", "avg_word_length",
        "link_count", "words_features")) 
      .setOutputCol("features")

    // TODO: add parameters?
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("score_double");
    
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, cvModel, processor, assembler, lr));
    val model = pipeline.fit(train);
    
    model.transform(test).select("body", "features", "score", "prediction").show();
  }
}
