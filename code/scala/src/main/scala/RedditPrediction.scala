package redditprediction

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext._
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{CommentBucketizerModel, CountVectorizerModel}

// import scopt._

case class Config(local_file: String = "", remote_file: String = "", sentiment: Boolean = false, limited: Boolean = true, mode: String = "linear");

object RedditPrediction {


  def main(args: Array[String]) {

    val parser = new scopt.OptionParser[Config]("scopt") {
      head("predict reddit", "1.0")
      opt[String]('l', "local_file") action { (x, c) =>
        c.copy(local_file = x) } text("specify the full path of a local json file")
      opt[String]('r', "remote_file") action { (x, c) =>
        c.copy(remote_file = x) } text("subreddit name for s3 bucket access")
      opt[String]('m', "mode") action { (x, c) =>
        c.copy(mode = x) } text("algorithm mode")
      opt[Unit]('s', "sentiment") action { (_, c) =>
        c.copy(sentiment = true) } text("only use sentiment words")
      opt[Unit]('u', "unlimited") action { (_, c) =>
        c.copy(limited = false) } text("remove limit (for cluster jobs")
      help("help") text("prints this usage text")
    }

    var input_file: String = ""
    var mode : String = ""
    var sentiment: Boolean = false
    var to_limit: Boolean = true
    parser.parse(args, Config()) match { 
      case Some(config) => {
        if (config.remote_file.length > 0) {
          input_file = "s3n://cs260-yoavz/" + config.remote_file + ".json"
        } else if (config.local_file.length > 0) {
          input_file = config.local_file
        } else {
          println("Must specify a local or remote file!");
          return
        }

        mode = config.mode;
        sentiment = config.sentiment;
        to_limit = config.limited;
      }

      case None => {
        println("argument error")
        return
      }
    }

    val conf = new SparkConf().setAppName("Predict Reddit Comments")
    val sc = new SparkContext()
    sc.hadoopConfiguration.set("fs.s3.awsAccessKeyId", sys.env("AWS_ACCESS_KEY_ID"))
    sc.hadoopConfiguration.set("fs.s3.awsSecretAccessKey", sys.env("AWS_SECRET_ACCESS_KEY"))
    val sqlContext = new SQLContext(sc);

    println(s"Loading from ${input_file}")
    val df = sqlContext.read.json(input_file);
    println(s"${df.count()} total comments");

    // Filtering logic for removed and deleted comments
    val filtered = df.filter(df("author") !== "[deleted]")
                     .filter(df("body") !== "[removed]")
                     .filter(df("body") !== "[deleted]");
    println(s"${filtered.count()} total comments after filtering");

    // Limiting
    val limited = if (to_limit) { 
      filtered.limit(5000)
    } else {
      filtered
    }
    println(s"${limited.count()} comments after limiting");

    // Preprocessing
    var featurePipeline: FeaturePipeline = new FeaturePipeline();
    if (sentiment) {
      featurePipeline = new SentimentFeaturePipeline();
    } 
    val featurePipelineModel: FeaturePipelineModel = featurePipeline.fit(limited)
    val featured = featurePipelineModel.transform(limited) 
    println(s"${featured.count()} total comments after preprocessing");

    // Split Train/Test
    val train_to_test = 0.70;
    val Array(train, test) = featured.randomSplit(Array(train_to_test, 1-train_to_test));
    println(s"Split into ${train.count()} training and ${test.count()} test comments");

    // Do the training
    if (mode == "logistic") {
      val regs: Array[Double] = (-5 to 5).toArray.map(x => scala.math.pow(2, x))
      val logistic = new RedditLogisticRegression();
      logistic.trainWithRegularization(train, regs)
      logistic.test(test)
      println("-") 
      println(s"Actual (training) distribution:")
      val bucketizer: CommentBucketizerModel = 
        featurePipelineModel.model.stages(4).asInstanceOf[CommentBucketizerModel]
      bucketizer.explainBucketDistribution(featured, "score_bucket")
    } else if (mode == "ridge") {
      val regs: Array[Double] = (-5 to 8).toArray.map(x => scala.math.pow(2, x))
      println("Learning using Ridge Regression");
      val regr = new RedditRidgeRegression();
      regr.train(train, regs);
      regr.test(test);
      featurePipelineModel.explainWeights(
        regr.getRegressionModel.weights.toArray, 10)
    } else {
      println("Learning using Vanilla Linear Regression");
      val regr = new RedditRegression();
      regr.train(train);
      regr.test(test);
      featurePipelineModel.explainWeights(
        regr.getRegressionModel.weights.toArray, 10)
    }
  }
}
