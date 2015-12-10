package redditprediction.pipeline

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{Tokenizer, VectorAssembler, OneHotEncoder}

class NaiveBayesFeaturePipeline(binary: Boolean = false) extends FeaturePipeline {
  override val bucketizer: CommentBucketizer = new CommentBucketizer(binary)
    .setScoreCol("score_double")
    .setScoreBucketCol("score_bucket")

  override val assembler = new VectorAssembler()
    .setInputCols(Array("words_features"))
    .setOutputCol("features")

  override def fit(dataset: DataFrame): NaiveBayesFeaturePipelineModel = {
    new NaiveBayesFeaturePipelineModel(getPipeline.fit(dataset))
  }
}

class NaiveBayesFeaturePipelineModel(modelc: PipelineModel) extends FeaturePipelineModel(modelc) {
  override def getCountVectorizerModel = null
}
