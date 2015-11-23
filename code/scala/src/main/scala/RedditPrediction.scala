import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext._
import org.apache.spark.sql.SQLContext

import redditprediction.RedditLinearRegression

case class Config(input: String = "")

object RedditPrediction {
  def main(args: Array[String]) {

    if (args.length < 1) {
      println("Please provide an input file argument");
      return;
    }

    val input_file = args(0);
    // val mode = args(1);

    val conf = new SparkConf().setAppName("Predict Reddit Comments")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc);

    val df = sqlContext.read.json(input_file);
    println(s"Loaded input file ${input_file}");
    println(s"${df.count()} total comments");

    // Filtering logic for removed and deleted comments
    val filtered = df.filter(df("author") !== "[deleted]")
                     .filter(df("body") !== "[removed]")
                     .filter(df("body") !== "[deleted]");
    println(s"${filtered.count()} total comments after filtering");

    val train_to_test = 0.6;
    val splits = filtered.randomSplit(Array(train_to_test, 1-train_to_test));
    val train = splits(0);
    val test = splits(1);
    println(s"Split into ${train.count()} training and ${test.count()} test comments");

    val regr = new RedditLinearRegression(train, test);
    regr.run();
  }
}
