package org.apache.spark.ml.feature

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf, max}
import org.apache.spark.sql.types.{StructType, StructField}

class CommentBucketizer(override val uid: String) 
  extends Estimator[CommentBucketizerModel] {
  def this() = this(Identifiable.randomUID("commentBucketizer"))

  val scoreBucketCol: Param[String] = new Param[String](this, "score_bucket", "");
  def setScoreBucketCol(value: String): this.type = set(scoreBucketCol, value)
  val scoreCol: Param[String] = new Param[String](this, "score", "");
  def setScoreCol(value: String): this.type = set(scoreCol, value)

  override def fit(dataset: DataFrame): CommentBucketizerModel = {
    transformSchema(dataset.schema, logging = true);

    val maxScore = dataset.select(max($(scoreCol))).head().getDouble(0)
    val model = new CommentBucketizerModel(maxScore)
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

class CommentBucketizerModel(override val uid: String, maxScoreA: Double)
  extends Model[CommentBucketizerModel] {

  def this(maxScore: Double) = 
    this(Identifiable.randomUID("commentBucketizerModel"), maxScore)

  val scoreBucketCol: Param[String] = new Param[String](this, "score_bucket", "");
  def setScoreBucketCol(value: String): this.type = set(scoreBucketCol, value)
  val scoreCol: Param[String] = new Param[String](this, "score", "");
  def setScoreCol(value: String): this.type = set(scoreCol, value)

  var maxScore: Double = maxScoreA;

  def getBucketRange(bucket: Int): (Int, Int) = {
    if (bucket <= 0) {
      (-1, -1)
    } else {
      (scala.math.pow(2, bucket-1).toInt, scala.math.pow(2, bucket).toInt)
    }
  }

  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema, logging = true);
    // Assign a score bucket of 0 to negative scores
    // Assign a score bucket of log(score) to others (base 2)
    val bucketize = udf { score: Double =>
      if (score <= 0.0) {
        0.0
      } else {
        ((scala.math.log(score) / scala.math.log(2)).toInt + 1).toDouble
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
    val that = new CommentBucketizerModel(uid, maxScore)
    copyValues(that, extra)
  }
}  
