package redditprediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel,
                                    Tokenizer, VectorAssembler,
                                    StopWordsRemover, OneHotEncoder}
import org.apache.spark.ml.feature.{WordRemover, CommentTransformer, CommentBucketizer}

class SentimentFeaturePipeline extends FeaturePipeline {
  // initialization
  override val tokenizer: Tokenizer = new Tokenizer()
    .setInputCol("body")
    .setOutputCol("raw_words");

  val remover: WordRemover = new WordRemover()
    .setInputCol("raw_words")
    .setOutputCol("words")
    .loadSentimentMap("/root/data/AFINN-111.txt")

  override val cv: CountVectorizer = new CountVectorizer()
    .setInputCol("words")
    .setOutputCol("words_features");

  override val processor: CommentTransformer = new CommentTransformer()
    .setWordsCol("words")
    .setBodyCol("body")
    .setScoreCol("score_double")
    .setTimeCol("created_utc")
    .setWordsCountCol("words_count")
    .setCharsCountCol("chars_count")
    .setAvgWordLengthCol("avg_word_length")
    .setLinkCountCol("link_count")
    .setHourCol("hour")
    .setSentimentCol("sentiment")
    .loadSentimentMap("/root/data/AFINN-111.txt")

  override val bucketizer: CommentBucketizer = new CommentBucketizer()
    .setScoreCol("score_double")
    .setScoreBucketCol("score_bucket")

  override val hourEncoder: OneHotEncoder = new OneHotEncoder()
    .setInputCol("hour")
    .setOutputCol("hour_encoded")

  override val assembler = new VectorAssembler()
    .setInputCols(Array("words_features", "chars_count", "avg_word_length",
      "link_count", "words_count", "hour_encoded", "sentiment")) 
    .setOutputCol("features")

  override def getPipeline: Pipeline = {
    new Pipeline().setStages(Array(tokenizer, remover, cv, 
                                   processor, bucketizer, 
                                   hourEncoder, assembler))
  }

  override def fit(dataset: DataFrame): FeaturePipelineModel = {
    new SentimentFeaturePipelineModel(getPipeline.fit(dataset))
  }
}

class SentimentFeaturePipelineModel(modelc: PipelineModel) extends FeaturePipelineModel(modelc) {
  override  def getCountVectorizerModel = {
    model.stages(2).asInstanceOf[CountVectorizerModel]; 
  }
}
