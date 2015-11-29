package redditprediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel,
                                    Tokenizer, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.classification.{OneVsRest, LogisticRegression}
import org.apache.spark.ml.feature.{CommentTransformer, CommentBucketizer}


class RedditLogisticRegression(val trainc: DataFrame, val testc: DataFrame) {

  var train: DataFrame = trainc;
  var test: DataFrame = testc;

  def run() {
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
      .setTimeCol("created_utc")
      .setWordsCountCol("words_count")
      .setCharsCountCol("chars_count")
      .setAvgWordLengthCol("avg_word_length")
      .setLinkCountCol("link_count")
      .setHourCol("hour")

    val hourEncoder: OneHotEncoder = new OneHotEncoder()
      .setInputCol("hour")
      .setOutputCol("hour_encoded")

    val assembler = new VectorAssembler()
      .setInputCols(Array("words_count", "chars_count", "avg_word_length",
        "link_count", "words_features", "hour_encoded")) 
      .setOutputCol("features")

    val featuresPipeline = new Pipeline()
      .setStages(Array(tokenizer, cvModel, processor, hourEncoder, assembler))

    val bucketizer: CommentBucketizer = new CommentBucketizer(0)
      .setScoreCol("score_double")
      .setScoreBucketCol("score_bucket")

    val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("score_bucket");

    val multiLr = new OneVsRest()
      .setClassifier(lr)
      .setFeaturesCol("features")
      .setLabelCol("score_bucket");

    val pipeline = new Pipeline()
      .setStages(Array(featuresPipeline, bucketizer, multiLr));

    val model = pipeline.fit(train);
    model.transform(test).select("body", "features", "score", "score_bucket", "prediction").show();
  }
}
