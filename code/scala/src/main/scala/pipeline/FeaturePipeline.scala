package redditprediction.pipeline

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{CountVectorizer, Tokenizer, VectorAssembler,
                                    StopWordsRemover, OneHotEncoder, 
                                    CountVectorizerModel, RegexTokenizer}

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

  def getCountVectorizerModel = {
    model.stages(2).asInstanceOf[CountVectorizerModel]; 
  }

  def explainWeights(weights: Array[Double], top: Int) = {
    println(s"Displaying top ${top} features");
    weights.zipWithIndex.sortBy(-_._1).take(top).foreach{ t =>
      val weight = t._1
      val idx = t._2
      if (getCountVectorizerModel != null &&
          idx < getCountVectorizerModel.vocabulary.length) {
        println(s"[${weight}] Word ${getCountVectorizerModel.vocabulary(idx)}");
      } else {
        println(s"[${weight}] Feature ${idx}");
      }
    }
  }
}
