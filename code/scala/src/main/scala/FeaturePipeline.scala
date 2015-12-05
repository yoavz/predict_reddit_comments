package redditprediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{CountVectorizer, Tokenizer, VectorAssembler,
                                    StopWordsRemover, OneHotEncoder, 
                                    CountVectorizerModel}
import org.apache.spark.ml.feature.{CommentTransformer, CommentBucketizer}

class FeaturePipeline {
  // initialization
  val tokenizer: Tokenizer = new Tokenizer()
    .setInputCol("body")
    .setOutputCol("words");

  val stopwords: StopWordsRemover = new StopWordsRemover()
    .setInputCol("words")
    .setOutputCol("filtered")

  val cv: CountVectorizer = new CountVectorizer()
    .setInputCol("filtered")
    .setOutputCol("words_features");

  val processor: CommentTransformer = new CommentTransformer()
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

  val hourEncoder: OneHotEncoder = new OneHotEncoder()
    .setInputCol("hour")
    .setOutputCol("hour_encoded")

  val bucketizer: CommentBucketizer = new CommentBucketizer()
    .setScoreCol("score_double")
    .setScoreBucketCol("score_bucket")

  val assembler = new VectorAssembler()
    .setInputCols(Array("words_features", "chars_count", "avg_word_length",
      "link_count", "words_count", "hour_encoded", "sentiment")) 
    .setOutputCol("features")


  def getPipeline: Pipeline = {
    new Pipeline().setStages(Array(tokenizer, stopwords, cv, 
                                   processor, bucketizer, 
                                   hourEncoder, assembler))
  }

  def fit(dataset: DataFrame): FeaturePipelineModel = {
    new FeaturePipelineModel(getPipeline.fit(dataset))
  }
}

class FeaturePipelineModel(modelc: PipelineModel) {
  val model: PipelineModel = modelc
  def transform(dataset: DataFrame): DataFrame = { model.transform(dataset) };
}
