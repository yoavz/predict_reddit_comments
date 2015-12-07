package redditprediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.functions._

class RedditRegression {

  var _model: Option[PipelineModel] = None
  def setModel(model: PipelineModel) = {
    _model = Some(model)
  }
  def getModel() = {
    _model match {
      case Some(value) => value
      case None => throw new RuntimeException("model not set")
    }
  }

  def getRegressionModel = {
    getModel.stages(0).asInstanceOf[LinearRegressionModel]
  }

  def train(dataset: DataFrame) = {
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("score_double")
      .setMaxIter(100);
    
    val pipeline = new Pipeline()
      .setStages(Array(lr));
    val model = pipeline.fit(dataset);
    setModel(model)

    val lrModel = model.stages(0).asInstanceOf[LinearRegressionModel]
    if (lrModel.hasSummary) {
      println(s"Training RMSE: ${lrModel.summary.rootMeanSquaredError}")
    }
  }

  def test(dataset: DataFrame) = {
    val predictions = getModel.transform(dataset);
    val error = udf { (score: Double, prediction: Double) =>
      scala.math.pow(score - prediction, 2) 
    }
    val sqError = predictions.withColumn("sq_err", 
                              error(col("score_double"), col("prediction")));
    val mse: Double = sqError.agg(avg(col("sq_err"))).first().getDouble(0)
    val rmse: Double = scala.math.sqrt(mse)

    println(s"Testing RMSE: ${rmse}");
    sqError.select("body", "words", "score_double", "prediction", "sq_err").show(10)
  }
}
