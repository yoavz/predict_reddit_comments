package redditprediction.pipeline

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel,
                                    Tokenizer, VectorAssembler,
                                    StopWordsRemover, OneHotEncoder}

class SentimentFeaturePipeline extends FeaturePipeline {
  // initialization
  override val tokenizer: Tokenizer = new Tokenizer()
    .setInputCol("body")
    .setOutputCol("raw_words");

  val remover: WordRemover = new WordRemover()
    .setInputCol("raw_words")
    .setOutputCol("words")
    .loadSentimentMap("/root/data/AFINN-111.txt")

  override val assembler = new VectorAssembler()
    .setInputCols(Array("words_features", "chars_count", "avg_word_length",
      "link_count", "words_count", "hour_encoded", "sentiment")) 
    .setOutputCol("features")

  override def getPipeline: Pipeline = {
    new Pipeline().setStages(Array(tokenizer, remover, cv, 
                                   processor, bucketizer, 
                                   hourEncoder, assembler))
  }

  override def fit(dataset: DataFrame): SentimentFeaturePipelineModel = {
    new SentimentFeaturePipelineModel(getPipeline.fit(dataset))
  }
}

class SentimentFeaturePipelineModel(modelc: PipelineModel) extends FeaturePipelineModel(modelc) {
  override  def getCountVectorizerModel = {
    model.stages(2).asInstanceOf[CountVectorizerModel]; 
  }
}
