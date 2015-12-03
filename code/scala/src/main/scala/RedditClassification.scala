package redditprediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, Model, PipelineModel}
import org.apache.spark.ml.classification.{Classifier, OneVsRest, OneVsRestModel, LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature.{CountVectorizerModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import org.apache.spark.ml.feature.{CommentBucketizer, CommentBucketizerModel}

abstract class RedditClassification {

  var _model: Option[PipelineModel] = None
  def setModel(model: PipelineModel) = {
    _model = Some(model)
  }
  def getModel() = {
    _model match {
      case Some(value) => value
      case None => throw new RuntimeException("Model not set, cannot call getModel")
    }
  }

  // abstract methods
  def getClassifier: Classifier[_, _, _]
  def explainTraining = {
    println("explainTraining not defined");
  }

  def explainBucketDistribution(dataset: DataFrame, bucketCol: String) = {
    val model = getModel
    val bucketizerModel: CommentBucketizerModel =
      model.stages(1).asInstanceOf[CommentBucketizerModel];
    bucketizerModel.explainBucketDistribution(dataset, bucketCol)
  }

  def train(dataset: DataFrame) = {
    // Set up the pipeline
    val bucketizer: CommentBucketizer = new CommentBucketizer()
      .setScoreCol("score_double")
      .setScoreBucketCol("score_bucket")
    val lr = getClassifier
    val multiLr = new OneVsRest()
      .setClassifier(lr)
      .setFeaturesCol("features")
      .setLabelCol("score_bucket");
    val pipeline = new Pipeline()
      .setStages(Array(FeaturePipeline, bucketizer, multiLr));

    val model = pipeline.fit(dataset)
    setModel(model)
  }


  def test(dataset: DataFrame) = {
    val model = getModel
    // test accuracy
    val predictions = model.transform(dataset);
    val accuracy = predictions.filter("score_bucket = prediction")
                              .count().toDouble / predictions.count().toDouble
    println(s"Test Accuracy: ${accuracy}");
    println(s"-");
    println(s"Actual distribution:")
    explainBucketDistribution(predictions, "score_bucket")
    println(s"-");
    println(s"Prediction distribution:")
    explainBucketDistribution(predictions, "prediction")
  }

  // TODO: saving models may be hard :(
  // def saveModelLocal(filename: String) = {
  //   val pickle: String = getModel.pickle.value
  //   println(pickle);
  //   // val oos = new FileOutputStream(filename);
  //   // oos.write(pickle);
  //   // oos.close;
  // }
  //
  // def loadModelLocal(filename: String) = {
  //   // val ios = new FileInputStream(filename);
  //   // val model = ios.read.unpickle
  //   // setModel(model);
  // }
}
