package redditprediction.pipeline

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf, max, mean}
import org.apache.spark.sql.types.{StructType, StructField}

import org.apache.log4j.Logger
import org.apache.log4j.Level

class CommentBucketizer(override val uid: String, binary: Boolean = false) 
  extends Estimator[CommentBucketizerModel] {
  def this(binary: Boolean) = this(Identifiable.randomUID("commentBucketizer"), binary)

  val scoreBucketCol: Param[String] = new Param[String](this, "score_bucket", "");
  def setScoreBucketCol(value: String): this.type = set(scoreBucketCol, value)
  val scoreCol: Param[String] = new Param[String](this, "score", "");
  def setScoreCol(value: String): this.type = set(scoreCol, value)

  override def fit(dataset: DataFrame): CommentBucketizerModel = {
    transformSchema(dataset.schema, logging = true);

    val maxScore = dataset.select(max($(scoreCol))).head().getDouble(0)
    val count: Int = dataset.count().toInt
    val medIdx: Int = count - count / 4;
    val meanScore: Double = dataset.select($(scoreCol))
                                   .sort($(scoreCol))
                                   .take(medIdx)
                                   .last.getDouble(0)
    val model = new CommentBucketizerModel(maxScore, meanScore, binary)
    model.setScoreBucketCol($(scoreBucketCol))
    model.setScoreCol($(scoreCol))
    model
  }

  override def transformSchema(schema: StructType): StructType = {
    var bucket = NumericAttribute.defaultAttr.withName($(scoreBucketCol));
    val outputFields = schema.fields :+ bucket.toStructField()
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): CommentBucketizer = defaultCopy(extra)
}

class CommentBucketizerModel(override val uid: String, maxScoreA: Double, meanScoreA: Double, binaryA: Boolean)
  extends Model[CommentBucketizerModel] {

  def this(maxScore: Double, meanScore: Double, binary: Boolean) = 
    this(Identifiable.randomUID("commentBucketizerModel"), maxScore, meanScore, binary)

  val scoreBucketCol: Param[String] = new Param[String](this, "score_bucket", "");
  def setScoreBucketCol(value: String): this.type = set(scoreBucketCol, value)
  val scoreCol: Param[String] = new Param[String](this, "score", "");
  def setScoreCol(value: String): this.type = set(scoreCol, value)

  var maxScore: Double = maxScoreA;
  var meanScore: Double = meanScoreA;
  var binary: Boolean = binaryA;

  def getBucketRange(bucket: Int): (Int, Int) = {
    if (bucket <= 0) {
      (-1, -1)
    } else {
      (scala.math.pow(2, bucket-1).toInt, scala.math.pow(2, bucket).toInt)
    }
  }

  def explainBucketDistribution(dataset: DataFrame, bucketCol: String, log: Logger) = {
    val count = dataset.count()
    if (binary) {
      val count_below = dataset.filter(col("score_double").leq(meanScore)).count()
      val count_above = dataset.filter(col("score_double").gt(meanScore)).count()
      log.info(s"Median: ${meanScore} ")
      log.info(s"< Median: ${count_below} (${count_below.toDouble / count.toDouble})")
      log.info(s">= Median: ${count_above} (${count_above.toDouble / count.toDouble})")
    } else {
      val count = dataset.count()
      val buckets = dataset.groupBy(bucketCol).count()
                           .rdd.map(r => (r.getDouble(0), r.getLong(1)))
                           .collect().sortBy(_._1)
      buckets.foreach{ t =>
        val (bucket, bucket_count) = t
        log.info(s"""Score Bucket ${getBucketRange(bucket.toInt)}: 
                    |${bucket_count.toDouble / count.toDouble} 
                    |(${bucket_count.toDouble})""".stripMargin)
      } 
    }
  }

  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema, logging = true);
    // Assign a score bucket of 0 to negative scores
    // Assign a score bucket of log(score) to others (base 2)
    val bucketize = udf { score: Double =>
      if (binary) {
        if (score > meanScore) { 1 } else { 0 }
      } else {
        if (score <= 0.0) {
          0.0
        } else {
          ((scala.math.log(score) / scala.math.log(2)).toInt + 1).toDouble
        }
      }
    }
    dataset.withColumn($(scoreBucketCol), bucketize(col($(scoreCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    var bucket = NumericAttribute.defaultAttr.withName($(scoreBucketCol));
    val outputFields = schema.fields :+ bucket.toStructField()
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): CommentBucketizerModel = {
    val that = new CommentBucketizerModel(uid, maxScore, meanScore, binary)
    copyValues(that, extra)
  }
}  
