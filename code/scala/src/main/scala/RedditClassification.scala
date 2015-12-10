package redditprediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, Model, PipelineModel}
import org.apache.spark.ml.classification.{Classifier, OneVsRest, OneVsRestModel, LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature.{CountVectorizerModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

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

  def train(dataset: DataFrame) = {
    // Set up the pipeline
    val lr = getClassifier
    val multiLr = new OneVsRest()
      .setClassifier(lr)
      .setFeaturesCol("features")
      .setLabelCol("score_bucket");
    val pipeline = new Pipeline()
      .setStages(Array(multiLr));

    val model = pipeline.fit(dataset)
    setModel(model)
  }

  def test(dataset: DataFrame): Double = {
    val model = getModel
    // test accuracy
    val predictions = model.transform(dataset);
    val accuracy = predictions.filter("score_bucket = prediction")
                              .count().toDouble / predictions.count().toDouble
    return accuracy
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
