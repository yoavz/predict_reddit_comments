package redditprediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.RegressionEvaluator

class RedditRidgeRegressionTFIDF(val trainc: DataFrame, val testc: DataFrame) {

  var train: DataFrame = trainc;
  var test: DataFrame = testc;

  def run() {
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("score_double");
    
    val pipeline = new Pipeline()
      .setStages(Array(lr));

    val evaluator = new RegressionEvaluator()
      .setLabelCol("score_double")
      .setPredictionCol("prediction");

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0))
      .build();

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8);

    val model = trainValidationSplit.fit(train);
    model.transform(test).select("body", "features", "score", "prediction").show();
  }
}
