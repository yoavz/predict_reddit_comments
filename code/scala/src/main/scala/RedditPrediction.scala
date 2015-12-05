package redditprediction

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext._
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{CommentBucketizerModel}

object RedditPrediction {
  def main(args: Array[String]) {

    if (args.length < 1) {
      println("Please provide an input file argument");
      return;
    }

    val input_file = args(0);
    var mode = "";
    if (args.length > 1) {
      mode = args(1)
    } 
    var features = ""
    if (args.length > 2) {
      features = args(2)
    }

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

    // Preprocessing
    var featurePipeline: FeaturePipeline = new FeaturePipeline();
    if (features == "sentiment") {
      featurePipeline = new SentimentFeaturePipeline();
    } 
    val featurePipelineModel: FeaturePipelineModel = featurePipeline.fit(filtered)
    val featured = featurePipelineModel.transform(filtered) 
    println(s"${featured.count()} total comments after preprocessing");

    // Split Train/Test
    val train_to_test = 0.9;
    val Array(train, test) = featured.randomSplit(Array(train_to_test, 1-train_to_test));
    println(s"Split into ${train.count()} training and ${test.count()} test comments");
    val regs: Array[Double] = (-5 to 5).toArray.map(x => scala.math.pow(2, x))

    // Do the training
    if (mode == "logistic") {
      val logistic = new RedditLogisticRegression();
      logistic.trainWithRegularization(train, regs)
      logistic.test(test)
      println("-") 
      println(s"Actual (training) distribution:")
      val bucketizer: CommentBucketizerModel = 
        featurePipelineModel.model.stages(4).asInstanceOf[CommentBucketizerModel]
      bucketizer.explainBucketDistribution(featured, "score_bucket")
    } else if (mode == "ridge") {
      println("Learning using Ridge Regression");
      val regr = new RedditRidgeRegression();
      regr.train(train, regs);
      regr.test(test);
    } else {
      println("Learning using Vanilla Linear Regression");
      val regr = new RedditRegression();
      regr.train(train);
      regr.test(test);
    }
  }
}
