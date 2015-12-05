package redditprediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.regression.LinearRegression
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

  def train(dataset: DataFrame) = {
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("score_double");
    
    val pipeline = new Pipeline()
      .setStages(Array(lr));
    val model = pipeline.fit(dataset);
    setModel(model)
  }

  def test(dataset: DataFrame) = {
    val predictions = getModel.transform(dataset);
    val error = udf { (score: Double, prediction: Double) =>
      scala.math.pow(score - prediction, 2) 
    }
    val sqError = predictions.withColumn("sq_err", 
                              error(col("score_double"), col("prediction")));
    val rmse = sqError.agg(avg(col("sq_err"))).show()

    sqError.select("body", "score_double", "prediction", "sq_err").show(10)
  }
}
