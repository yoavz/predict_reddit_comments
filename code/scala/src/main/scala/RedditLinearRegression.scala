package redditprediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel,
                                    Tokenizer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.CommentTransformer

class RedditLinearRegression(val trainc: DataFrame, val testc: DataFrame) {

  var train: DataFrame = trainc;
  var test: DataFrame = testc;

  def run() {
    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCol("body")
      .setOutputCol("words");

    val cvModel: CountVectorizer = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("words_features");

    val processor: CommentTransformer = new CommentTransformer()
      .setWordsCol("words")
      .setBodyCol("body")
      .setScoreCol("score_double")
      .setWordsCountCol("words_count")
      .setCharsCountCol("chars_count")
      .setAvgWordLengthCol("avg_word_length")
      .setLinkCountCol("link_count")

    val assembler = new VectorAssembler()
      .setInputCols(Array("words_count", "chars_count", "avg_word_length",
        "link_count", "words_features")) 
      .setOutputCol("features")

    // TODO: add parameters?
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("score_double");
    
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, cvModel, processor, assembler, lr));
    val model = pipeline.fit(train);
    
    model.transform(test).select("body", "features", "score", "prediction").show();
  }
}
