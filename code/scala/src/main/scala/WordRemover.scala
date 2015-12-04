package org.apache.spark.ml.feature

import scala.io.Source

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._

class WordRemover(override val uid: String)
  extends Transformer with HasInputCol with HasOutputCol {

  def this() = this(Identifiable.randomUID("wordRemover"))

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

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: DataFrame): DataFrame = {
    val remove = udf { terms: Seq[String] => terms.filter(sentimentMap.contains) }
    val notEmpty = udf { terms: Seq[String] => if (terms.length > 0) true else false }
    val df = dataset.withColumn($(outputCol), remove(col($(inputCol))))
    val df2 = df.where(notEmpty(col($(outputCol))))
    df2
  }

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.appendColumn(schema, $(outputCol), new ArrayType(StringType, true))
  }

  override def copy(extra: ParamMap): CommentTransformer = defaultCopy(extra)
}
