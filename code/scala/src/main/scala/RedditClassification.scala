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

  // var _featurePipeline: Option[Pipeline] = None
  // def setFeaturePipeline(pipeline: Pipeline) = {
  //   _featurePipeline = Some(pipeline)
  // }
  // def getFeaturePipeline() = {
  //   _featurePipeline match {
  //     case Some(value) => value
  //     case None => throw new RuntimeException("Pipeline not set.")
  //   }
  // }
  //
  // abstract methods
  def getClassifier: Classifier[_, _, _]
  def explainTraining = {
    println("explainTraining not defined");
  }

  // def explainBucketDistribution(dataset: DataFrame, bucketCol: String) = {
  //   val model = getModel
  //   val bucketizerModel: CommentBucketizerModel =
  //     model.stages(1).asInstanceOf[CommentBucketizerModel];
  //   bucketizerModel.explainBucketDistribution(dataset, bucketCol)
  // }

  def train(dataset: DataFrame) = {
    // Set up the pipeline
    // TODO: move bucketizer to feature pipeline?
    val bucketizer: CommentBucketizer = new CommentBucketizer()
      .setScoreCol("score_double")
      .setScoreBucketCol("score_bucket")
    val lr = getClassifier
    val multiLr = new OneVsRest()
      .setClassifier(lr)
      .setFeaturesCol("features")
      .setLabelCol("score_bucket");
    val pipeline = new Pipeline()
      .setStages(Array(bucketizer, multiLr));

    val model = pipeline.fit(dataset)
    setModel(model)
  }

  def getBucketizer: CommentBucketizerModel = {
    getModel.stages(0).asInstanceOf[CommentBucketizerModel]
  }

  def test(dataset: DataFrame) = {
    val model = getModel
    // test accuracy
    val predictions = model.transform(dataset);
    val accuracy = predictions.filter("score_bucket = prediction")
                              .count().toDouble / predictions.count().toDouble
    println(s"Test Accuracy: ${accuracy}");
    // println(s"-");
    // println(s"Actual distribution:")
    // explainBucketDistribution(predictions, "score_bucket")
    // println(s"-");
    // println(s"Prediction distribution:")
    // explainBucketDistribution(predictions, "prediction")
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
