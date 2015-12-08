package redditprediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.regression.{RandomForestRegressor, 
                                       RandomForestRegressionModel}

class RedditRandomForestRegression extends RedditRegression {

  override def train(dataset: DataFrame) = {
    val forest = new RandomForestRegressor()
      .setFeaturesCol("features")
      .setLabelCol("score_double");
    
    val pipeline = new Pipeline()
      .setStages(Array(forest));

    val model = pipeline.fit(dataset)
    setModel(model)

    val forestModel = model.stages(0).asInstanceOf[RandomForestRegressionModel]
    println(s"Random Forest Summary: ${forestModel.toString}")
  }
}
