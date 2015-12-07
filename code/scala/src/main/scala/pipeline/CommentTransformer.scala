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

  // input cols
  val bodyCol: String = "body";
  val wordsCol: String = "words";
  val timeCol: String = "created_utc";

  // output cols
  val scoreCol: String = "score_double";
  val wordsCountCol: String = "words_count"
  val charsCountCol: String = "chars_count"
  val avgWordLengthCol: String = "avg_word_length"
  val linkCountCol: String = "link_count"
  val hourCol: String = "hour"
  val sentimentCol: String = "sentiment"
  val questionMarkCol: String = "question_mark"
  val exclMarkCol: String = "excl_mark"

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
    // set param names

    transformSchema(dataset.schema, logging = true);

    // TODO: find a cleaner way to apply a sequence of udfs to columns?
    // cast the scores to integers
   
    // Word/character counts
    val scores = dataset.withColumn(scoreCol, col("score").cast(DoubleType))
    val countWords = udf { terms: Seq[String] => terms.length }
    var df = scores.withColumn(wordsCountCol, countWords(col(wordsCol)));
    val countChars = udf { terms: Seq[String] => 
      terms.foldLeft(0)((z: Int, i: String) => z + i.length() )
    }
    df = df.withColumn(charsCountCol, countChars(col(wordsCol)));
    val avgWordLength = udf { terms: Seq[String] =>
      terms.foldLeft(0)((z: Int, i: String) => z + i.length()).toDouble / terms.length;
    }
    df = df.withColumn(avgWordLengthCol, avgWordLength(col(wordsCol)));

    // Count links
    val countLinks = udf { body: String => LINK_RE.findAllIn(body).size }
    df = df.withColumn(linkCountCol, countLinks(col(bodyCol)));

    // Hour of day
    val timeHour = udf { time: String =>
      val created: DateTime = new DateTime(time.toLong * 1000L)
      created.getHourOfDay().toDouble
    }
    df = df.withColumn(hourCol, timeHour(col(timeCol)))

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
    df = df.withColumn(sentimentCol, sentimentAnalaysis(col(wordsCol)))

    // Metadata punc counts
    def countChar(char: Char) = { udf { body: String => body.count(_ == char) } }
    val questMarkCount = countChar('?')
    val exclMarkCount = countChar('!')

    df = df.withColumn(questionMarkCol, questMarkCount(col(bodyCol)))
    df = df.withColumn(exclMarkCol, exclMarkCount(col(bodyCol)))

    df
  }

  override def transformSchema(schema: StructType): StructType = {

    // new columns
    var numWords = NumericAttribute.defaultAttr.withName(wordsCountCol);
    var numChars = NumericAttribute.defaultAttr.withName(charsCountCol);
    var avgWordLength = NumericAttribute.defaultAttr.withName(avgWordLengthCol);
    var numLinks = NumericAttribute.defaultAttr.withName(linkCountCol);
    var scores = NumericAttribute.defaultAttr.withName(scoreCol);
    var sentiment = NumericAttribute.defaultAttr.withName(sentimentCol);
    var hour = NominalAttribute.defaultAttr.withName(hourCol).withNumValues(24)

    var quest = NumericAttribute.defaultAttr.withName(questionMarkCol);
    var excl = NumericAttribute.defaultAttr.withName(exclMarkCol)

    val outputFields = schema.fields :+ numWords.toStructField() :+
      numChars.toStructField() :+ avgWordLength.toStructField() :+
      numLinks.toStructField() :+ scores.toStructField() :+ 
      hour.toStructField() :+ sentiment.toStructField() :+
      quest.toStructField() :+ excl.toStructField() ;
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): CommentTransformer = defaultCopy(extra)
}

