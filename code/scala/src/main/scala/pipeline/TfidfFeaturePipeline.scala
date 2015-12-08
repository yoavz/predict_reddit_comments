package redditprediction.pipeline

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{VectorAssembler, HashingTF, IDF}

class TfidfFeaturePipeline extends FeaturePipeline {

  val hashingTF = new HashingTF()
    .setInputCol("words")
    .setOutputCol("tf")
    .setNumFeatures(10000)

  val idf = new IDF()
    .setInputCol("tf")
    .setOutputCol("words_features")

  override def getPipeline: Pipeline = {
    new Pipeline().setStages(Array(tokenizer, hashingTF, idf, 
                                   processor, bucketizer, 
                                   hourEncoder, assembler))
  }

  override def fit(dataset: DataFrame): TfidfFeaturePipelineModel = {
    new TfidfFeaturePipelineModel(getPipeline.fit(dataset))
  }
}

class TfidfFeaturePipelineModel(modelc: PipelineModel) extends FeaturePipelineModel(modelc) {
  override def getCountVectorizerModel = null
}
