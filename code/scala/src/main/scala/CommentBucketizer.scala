package org.apache.spark.ml.feature

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf, max}
import org.apache.spark.sql.types.{StructType, StructField}

class CommentBucketizer(override val uid: String, numBucketsA: Int) 
  extends Estimator[CommentBucketizerModel] {
  def this(numBuckets: Int) = 
    this(Identifiable.randomUID("commentBucketizer"), numBuckets)

  val scoreBucketCol: Param[String] = new Param[String](this, "score_bucket", "");
  def setScoreBucketCol(value: String): this.type = set(scoreBucketCol, value)
  val scoreCol: Param[String] = new Param[String](this, "score", "");
  def setScoreCol(value: String): this.type = set(scoreCol, value)

  val numBuckets: Int = numBucketsA;

  override def fit(dataset: DataFrame): CommentBucketizerModel = {
    transformSchema(dataset.schema, logging = true);

    val maxScore = dataset.select(max($(scoreCol))).head().getDouble(0)
    val model = new CommentBucketizerModel(numBuckets, maxScore)
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

class CommentBucketizerModel(override val uid: String, 
                             numBucketsA: Int, maxScoreA: Double)
  extends Model[CommentBucketizerModel] {

  def this(numBuckets: Int, maxScore: Double) = 
    this(Identifiable.randomUID("commentBucketizerModel"), numBuckets, maxScore)

  val scoreBucketCol: Param[String] = new Param[String](this, "score_bucket", "");
  def setScoreBucketCol(value: String): this.type = set(scoreBucketCol, value)
  val scoreCol: Param[String] = new Param[String](this, "score", "");
  def setScoreCol(value: String): this.type = set(scoreCol, value)

  var maxScore: Double = maxScoreA;
  var numBuckets: Int = numBucketsA;

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

  override def copy(extra: ParamMap): CommentBucketizerModel = defaultCopy(extra)
}  
