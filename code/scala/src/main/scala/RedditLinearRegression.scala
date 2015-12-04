package redditprediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.LinearRegression

class RedditLinearRegression(val trainc: DataFrame, val testc: DataFrame) {

  var train: DataFrame = trainc;
  var test: DataFrame = testc;

  def run() {
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("score_double");
    
    val pipeline = new Pipeline()
      .setStages(Array(lr));
    val model = pipeline.fit(train);
    
    model.transform(test).select("body", "features", "score", "prediction").show();
  }
}
