package redditprediction.pipeline

import scala.io.Source

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{DoubleType, StructType, StructField}

import org.joda.time.DateTime

class CommentTransformer(override val uid: String)
  extends Transformer {
  
  val LINK_RE = """https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)""".r

  def this() = this(Identifiable.randomUID("commentTransformer"));

  // Input column: Seq of tokenized words
  val bodyCol: Param[String] = new Param[String](this, "body", "");
  val wordsCol: Param[String] = new Param[String](this, "words", "");
  val scoreCol: Param[String] = new Param[String](this, "score", "");
  val timeCol: Param[String] = new Param[String](this, "time", "");

  def setBodyCol(value: String): this.type = set(bodyCol, value)
  def setWordsCol(value: String): this.type = set(wordsCol, value)
  def setScoreCol(value: String): this.type = set(scoreCol, value)
  def setTimeCol(value: String): this.type = set(timeCol, value)

  // Output columns
  val wordsCountCol: Param[String] = new Param[String](this, "words_count", "");
  val charsCountCol: Param[String] = new Param[String](this, "chars_count", "");
  val avgWordLengthCol: Param[String] = new Param[String](this, "avg_word_length", "");
  val linkCountCol: Param[String] = new Param[String](this, "link_count", "");
  val hourCol: Param[String] = new Param[String](this, "hour", "");
  val sentimentCol: Param[String] = new Param[String](this, "sentiment", "");

  def setWordsCountCol(value: String): this.type = set(wordsCountCol, value)
  def setCharsCountCol(value: String): this.type = set(charsCountCol, value)
  def setAvgWordLengthCol(value: String): this.type = set(avgWordLengthCol, value)
  def setLinkCountCol(value: String): this.type = set(linkCountCol, value)
  def setHourCol(value: String): this.type = set(hourCol, value)
  def setSentimentCol(value: String): this.type = set(sentimentCol, value)

  // Sentiment Map
  var sentimentMap: Map[String, Int] = Map()
  def loadSentimentMap(filename: String) = {
    Source.fromFile(filename).getLines.foreach({ line:String  =>
      val tokens = line.split("\t")
      if (tokens.length == 2)
        sentimentMap += (tokens(0) -> tokens(1).toInt)
    })
    this
  }

  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema, logging = true);

    // TODO: find a cleaner way to apply a sequence of udfs to columns?
    // cast the scores to integers
   
    // Word/character counts
    val scores = dataset.withColumn($(scoreCol), col("score").cast(DoubleType))
    val countWords = udf { terms: Seq[String] => terms.length }
    val df = scores.withColumn($(wordsCountCol), countWords(col($(wordsCol))));
    val countChars = udf { terms: Seq[String] => 
      terms.foldLeft(0)((z: Int, i: String) => z + i.length() )
    }
    val df2 = df.withColumn($(charsCountCol), countChars(col($(wordsCol))));
    val avgWordLength = udf { terms: Seq[String] =>
      terms.foldLeft(0)((z: Int, i: String) => z + i.length()).toDouble / terms.length;
    }
    val df3 = df2.withColumn($(avgWordLengthCol), avgWordLength(col($(wordsCol))));

    // Count links
    val countLinks = udf { body: String => LINK_RE.findAllIn(body).size }
    val df4 = df3.withColumn($(linkCountCol), countLinks(col($(bodyCol))));

    // Hour of day
    val timeHour = udf { time: String =>
      val created: DateTime = new DateTime(time.toLong * 1000L)
      created.getHourOfDay().toDouble
    }
    val df5 = df4.withColumn($(hourCol), timeHour(col($(timeCol))))

    // Sentiment analysis
    val sentimentAnalaysis = udf { terms: Seq[String] =>
      var sentiment: Double = 0;
      var words: Int = 0;
      terms.foreach({ term: String => 
        if (sentimentMap.contains(term)) {
          sentiment += sentimentMap(term)
          words += 1
        }
      })
      if (words > 0) {
        sentiment / words
      } else {
        0
      }
    }
    df5.withColumn($(sentimentCol), sentimentAnalaysis(col($(wordsCol))))
  }

  override def transformSchema(schema: StructType): StructType = {

    // new columns
    var numWords = NumericAttribute.defaultAttr.withName($(wordsCountCol));
    var numChars = NumericAttribute.defaultAttr.withName($(charsCountCol));
    var avgWordLength = NumericAttribute.defaultAttr.withName($(avgWordLengthCol));
    var numLinks = NumericAttribute.defaultAttr.withName($(linkCountCol));
    var scores = NumericAttribute.defaultAttr.withName($(scoreCol));
    var sentiment = NumericAttribute.defaultAttr.withName($(sentimentCol));
    var hour = NominalAttribute.defaultAttr.withName($(hourCol)).withNumValues(24)

    val outputFields = schema.fields :+ numWords.toStructField() :+
      numChars.toStructField() :+ avgWordLength.toStructField() :+
      numLinks.toStructField() :+ scores.toStructField() :+ 
      hour.toStructField() :+ sentiment.toStructField();
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): CommentTransformer = defaultCopy(extra)
}

