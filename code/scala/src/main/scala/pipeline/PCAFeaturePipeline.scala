package redditprediction.pipeline

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{PCA, VectorAssembler, CountVectorizer}

class PCAFeaturePipeline(kc: Int) extends FeaturePipeline {

  val kDims: Int = kc; 

  // Vocab size must be < 60000 for PCA
  override val cv: CountVectorizer = new CountVectorizer()
    .setInputCol("filtered")
    .setOutputCol("words_features")
    .setVocabSize( 50000 );

  val pca = new PCA()
    .setInputCol("words_features")
    .setOutputCol("words_features_reduced")
    .setK(kDims)

  override val assembler = new VectorAssembler()
    .setInputCols(Array("words_features_reduced", "chars_count", "avg_word_length",
      "link_count", "words_count", "hour_encoded", "sentiment", "question_mark",
      "excl_mark")) 
    .setOutputCol("features")

  override def getPipeline: Pipeline = {
    new Pipeline().setStages(Array(tokenizer, tokenCleaner, stopwords, cv, 
                                   pca, processor, bucketizer, 
                                   hourEncoder, assembler))
  }

  override def fit(dataset: DataFrame): PCAFeaturePipelineModel = {
    new PCAFeaturePipelineModel(getPipeline.fit(dataset))
  }
}

class PCAFeaturePipelineModel(modelc: PipelineModel) extends FeaturePipelineModel(modelc) {
  override def getCountVectorizerModel = null
}
