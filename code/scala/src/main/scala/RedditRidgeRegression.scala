package redditprediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.RegressionEvaluator

class RedditRidgeRegression extends RedditRegression {

  def train(dataset: DataFrame, regs: Array[Double]): (Double, Double) = {
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("score_double");
    
    val pipeline = new Pipeline()
      .setStages(Array(lr));

    val evaluator = new RegressionEvaluator()
      .setLabelCol("score_double")
      .setPredictionCol("prediction");

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, regs)
      .build();

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8);

    val allModels = trainValidationSplit.fit(dataset);
    val model = allModels.bestModel.asInstanceOf[PipelineModel];
    setModel(model)

    println(s"Tried C: ${regs.deep.mkString(", ")}");
    val bestLr = model.stages(0).asInstanceOf[LinearRegressionModel]
    println(s"Best C: ${bestLr.getRegParam}")

    if (bestLr.hasSummary) {
      println(s"Training RMSE: ${bestLr.summary.rootMeanSquaredError}")
    }

    return (bestLr.getRegParam, bestLr.summary.rootMeanSquaredError)
  }
}
