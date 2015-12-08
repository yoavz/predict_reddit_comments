package redditprediction.pipeline

import scala.io.Source

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._

class WordCleaner(override val uid: String) extends Transformer { 

  def this() = this(Identifiable.randomUID("wordCleaner"))
  val inputCol: Param[String] = new Param[String](this, "input", "");
  val outputCol: Param[String] = new Param[String](this, "output", "");

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: DataFrame): DataFrame = {
    val clean = udf { terms: Seq[String] => 
      terms.map(t => t.replaceAll("[^A-Za-z0-9]", ""))
    }
    dataset.withColumn($(outputCol), clean(col($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    val outputFields = schema.fields :+ 
      StructField($(outputCol), new ArrayType(StringType, true), nullable = false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): CommentTransformer = defaultCopy(extra)
}
