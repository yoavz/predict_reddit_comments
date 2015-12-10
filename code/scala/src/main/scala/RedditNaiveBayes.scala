package redditprediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, Model, PipelineModel}
import org.apache.spark.ml.classification.{Classifier, OneVsRest, OneVsRestModel, NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

class RedditNaiveBayes(binary: Boolean) extends RedditClassification {

  override def getClassifier: NaiveBayes = {
    new NaiveBayes()
      .setFeaturesCol("features")
      .setLabelCol("score_bucket")
      .setModelType(if (binary) { "multinomial" } else { "multinomial" } );
  }

  override def train(dataset: DataFrame) = {
    // Set up the pipeline
    val bayes = getClassifier
    val pipeline = new Pipeline()
      .setStages(Array(bayes));

    val model = pipeline.fit(dataset)
    setModel(model)
  }

  def trainWithRegularization(dataset: DataFrame, regs: Array[Double]) = {
    println("not implemented")
    // // Set up the pipeline
    // val lr = getClassifier
    // val pipeline = new Pipeline()
    //   .setStages(Array(lr));
    //
    // // Evaluation and tuning
    // val evaluator = new MulticlassClassificationEvaluator()
    //   .setLabelCol("score_bucket")
    //   .setPredictionCol("prediction")
    // val paramGrid = new ParamGridBuilder()
    //   .addGrid(lr.regParam, regs)
    //   .build();
    // val trainValidationSplit = new TrainValidationSplit()
    //   .setEstimator(pipeline)
    //   .setEvaluator(evaluator)
    //   .setEstimatorParamMaps(paramGrid)
    //   .setTrainRatio(0.8);
    //
    // println("Training and tuning..."); 
    // val allModels = trainValidationSplit.fit(dataset);
    // val model = allModels.bestModel.asInstanceOf[PipelineModel];
    // setModel(model)
    //
    // println(s"Tried C: ${regs.deep.mkString(", ")}");
    // val bestLr = model.stages(0).asInstanceOf[OneVsRestModel]
    //                   .models(0).asInstanceOf[NaiveBayesModel];
    // println(s"Best C: ${bestLr.getRegParam}")
  }
}
