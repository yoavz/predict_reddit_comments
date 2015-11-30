package redditprediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{OneVsRest, LogisticRegression}
import org.apache.spark.ml.feature.CommentBucketizer

import redditprediction.FeaturePipeline

class RedditLogisticRegression(val trainc: DataFrame, val testc: DataFrame) {

  var train: DataFrame = trainc;
  var test: DataFrame = testc;

  def run() {
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
      .setStages(Array(FeaturePipeline, bucketizer, multiLr));

    val model = pipeline.fit(train);
    // model.transform(test).select("body", "features", "score", "score_bucket", "prediction").show();
    model.transform(test).show()
  }
}
