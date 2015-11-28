package org.apache.spark.ml.feature

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf, array}
import org.apache.spark.sql.types.{DoubleType, StructType, StructField, IntegerType}

class CommentBucketizer(override val uid: String)
  extends Estimator[CommentBucketizerModel] {
  def this() = this(Identifiable.randomUID("commentBucketizer"))
  val scoreBucketCol: Param[String] = new Param[String](this, "score_bucket", "");
  def setScoreBucketCol(value: String): this.type = set(scoreBucketCol, value)

  override def fit(dataset: DataFrame): CommentBucketizerModel = {
    transformSchema(dataset.schema, logging = true);

    val model = new CommentBucketizerModel(uid)
    model.setScoreBucketCol($(scoreBucketCol))
    model
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): CommentBucketizer = defaultCopy(extra)
}

class CommentBucketizerModel(override val uid: String)
  extends Model[CommentBucketizerModel] {

  def this() = this(Identifiable.randomUID("commentBucketizerModel"))
  val scoreBucketCol: Param[String] = new Param[String](this, "score_bucket", "");
  def setScoreBucketCol(value: String): this.type = set(scoreBucketCol, value)

  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema, logging = true);
    dataset
  }

  override def transformSchema(schema: StructType): StructType = {
    var bucket = NumericAttribute.defaultAttr.withName($(scoreBucketCol));
    val outputFields = schema.fields :+ bucket.toStructField()
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): CommentBucketizerModel = defaultCopy(extra)
}  
