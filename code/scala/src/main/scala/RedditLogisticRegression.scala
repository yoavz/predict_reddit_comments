package redditprediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{OneVsRest, OneVsRestModel, LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature.{CommentBucketizer, CountVectorizerModel}

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

    // recover the cvModel from the Feature Pipeline
    val cvModel: CountVectorizerModel = 
      model.stages(0).asInstanceOf[PipelineModel]
           .stages(1).asInstanceOf[CountVectorizerModel];

    // display the most important feautres
    model
      .stages(2).asInstanceOf[OneVsRestModel]
      .models.zipWithIndex
      .foreach{ case (c, i) =>
        println(s"Classifier bucket ${i}");
        val weights = c.asInstanceOf[LogisticRegressionModel].weights;
        weights
          .toArray.toList.zipWithIndex
          .sortWith((a, b) => a._1 > b._1).take(5)
          .foreach{ case (w, i) => 
            if (i < cvModel.vocabulary.length) {
              println(s"\t${w}\t${cvModel.vocabulary(i)}");
            } else {
              println(s"\t${w}\tother feature");
            }
          };
      };
  }
}
