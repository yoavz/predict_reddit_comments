package redditprediction.pipeline

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{Tokenizer, VectorAssembler, OneHotEncoder}

class MetadataFeaturePipeline extends FeaturePipeline {
  // initialization
  override val assembler = new VectorAssembler()
    .setInputCols(Array("chars_count", "avg_word_length",
      "link_count", "words_count", "hour_encoded", "sentiment",
      "question_mark", "excl_mark")) 
    .setOutputCol("features")


  override def getPipeline: Pipeline = {
    new Pipeline().setStages(Array(tokenizer, tokenCleaner, processor, bucketizer, 
                                   hourEncoder, assembler))
  }

  override def fit(dataset: DataFrame): MetadataFeaturePipelineModel = {
    new MetadataFeaturePipelineModel(getPipeline.fit(dataset))
  }
}

class MetadataFeaturePipelineModel(modelc: PipelineModel) extends FeaturePipelineModel(modelc) {
  override def getCountVectorizerModel = null

  override def getCommentBucketizerModel = {
    model.stages(3).asInstanceOf[CommentBucketizerModel]; 
  }
}
