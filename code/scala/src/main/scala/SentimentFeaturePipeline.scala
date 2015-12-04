package redditprediction

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{CountVectorizer, Tokenizer, VectorAssembler,
                                    StopWordsRemover, OneHotEncoder}
import org.apache.spark.ml.feature.{WordRemover, CommentTransformer}


class SentimentFeaturePipeline extends Pipeline {
  // initialization
  val tokenizer: Tokenizer = new Tokenizer()
    .setInputCol("body")
    .setOutputCol("raw_words");

  val remover: WordRemover = new WordRemover()
    .setInputCol("raw_words")
    .setOutputCol("words")
    .loadSentimentMap("/root/data/AFINN-111.txt")

  val cv: CountVectorizer = new CountVectorizer()
    .setInputCol("words")
    .setOutputCol("features");

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

  // val hourEncoder: OneHotEncoder = new OneHotEncoder()
  //   .setInputCol("hour")
  //   .setOutputCol("hour_encoded")

  // val assembler = new VectorAssembler()
  //   .setInputCols(Array("words_features", "chars_count", "avg_word_length",
  //     "link_count", "words_count", "hour_encoded", "sentiment")) 
    // .setOutputCol("features")

  this.setStages(Array(tokenizer, remover, cv, processor))
}
