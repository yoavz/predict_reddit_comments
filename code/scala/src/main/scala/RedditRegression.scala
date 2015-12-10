package redditprediction

import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.functions._

import org.apache.spark.mllib.linalg.Vector

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

  def train(dataset: DataFrame): Double = {
    train(dataset, 0.0)
  }

  def train(dataset: DataFrame, reg: Double): Double = {
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("score_double")
      .setRegParam(reg);
    
    val pipeline = new Pipeline()
      .setStages(Array(lr));
    val model = pipeline.fit(dataset);
    setModel(model)

    val lrModel = model.stages(0).asInstanceOf[LinearRegressionModel]

    return lrModel.summary.rootMeanSquaredError
  }

  def test(dataset: DataFrame): Double = {
    val predictions = getModel.transform(dataset);
    val error = udf { (score: Double, prediction: Double) =>
      scala.math.pow(score - prediction, 2)
    }
    val sqError = predictions.withColumn("sq_err", 
                              error(col("score_double"), col("prediction")));
    val mse: Double = sqError.agg(avg(col("sq_err"))).first().getDouble(0)
    val rmse: Double = scala.math.sqrt(mse)

    sqError.select("body", "words", "score_double", "prediction", "sq_err").show(10)
    return rmse
  }

  def trainSGD(train: DataFrame, test: DataFrame) = {
    def datasetToRDD = { dataset: DataFrame =>
      val scoreIdx = dataset.first().fieldIndex("score_double")
      val featuresIdx = dataset.first().fieldIndex("features")
      dataset.map{ row => 
        new org.apache.spark.mllib.regression.LabeledPoint(
          row.getDouble(scoreIdx),
          row.get(featuresIdx).asInstanceOf[Vector])
      }
    }

    val labeled_train = datasetToRDD(train)
    val labeled_test = datasetToRDD(test)

    labeled_train.take(10).foreach(println)

    val lr = new org.apache.spark.mllib.regression.LinearRegressionWithSGD();
    val model = lr.run(labeled_train)

    val predictions: RDD[Double] = model.predict(labeled_test.map(l => l.features))
    val actual: RDD[Double] = labeled_test.map(l => l.label)

    actual.take(10).foreach(println)
    predictions.take(10).foreach(println)
    // println(s"First 10 predictions: ${actual.take(10)}")
    // println(s"First 10 actual: ${predictions.take(10)}")

    val mse: Double = actual.zip[Double](predictions)
      .map(p => scala.math.pow(p._1 - p._2, 2))
      .mean()
    val rmse: Double = scala.math.sqrt(mse)
  }
}
