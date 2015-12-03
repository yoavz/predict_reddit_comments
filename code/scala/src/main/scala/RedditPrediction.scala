package redditprediction

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext._
import org.apache.spark.sql.SQLContext

object RedditPrediction {
  def main(args: Array[String]) {

    if (args.length < 1) {
      println("Please provide an input file argument");
      return;
    }

    val input_file = args(0);
    val mode = args(1);

    val conf = new SparkConf().setAppName("Predict Reddit Comments")
    val sc = new SparkContext()
    val sqlContext = new SQLContext(sc);

    val df = sqlContext.read.json(input_file);
    println(s"Loaded input file ${input_file}");
    println(s"${df.count()} total comments");

    // Filtering logic for removed and deleted comments
    val filtered = df.filter(df("author") !== "[deleted]")
                     .filter(df("body") !== "[removed]")
                     .filter(df("body") !== "[deleted]");
    println(s"${filtered.count()} total comments after filtering");

    val train_to_test = 0.9;
    val Array(train, test) = filtered.randomSplit(Array(train_to_test, 1-train_to_test));
    println(s"Split into ${train.count()} training and ${test.count()} test comments");
    
    if (mode == "logistic") {
      val logistic = RedditLogisticRegression;
      val regs: Array[Double] = (-5 to 5).toArray.map(x => scala.math.pow(2, x))
      logistic.trainWithRegularization(train, regs)
      logistic.test(test)
      logistic.explainTraining
    } else if (mode == "ridge") {
      println("Learning using Ridge Regression");
      val regr = new RedditRidgeRegression(train, test);
      regr.run();
    } else if (mode == "tfidf") {
      println("Learning using Ridge Regression w/ TF-IDF");
      val regr = new RedditRidgeRegressionTFIDF(train, test);
      regr.run();
    } else {
      println("Learning using Vanilla Linear Regression");
      val regr = new RedditLinearRegression(train, test);
      regr.run();
    }
  }
}
