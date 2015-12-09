package redditprediction

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{SQLContext, DataFrame, Row}
import org.apache.spark.sql.types.{StructType,StructField,StringType}

import com.google.cloud.hadoop.io.bigquery.BigQueryConfiguration
import com.google.cloud.hadoop.util.HadoopCredentialConfiguration;
import com.google.cloud.hadoop.io.bigquery.GsonBigQueryInputFormat
import com.google.gson.JsonObject
import org.apache.hadoop.io.{LongWritable, Text}

import redditprediction.pipeline.{CommentBucketizerModel,
                                  SentimentFeaturePipeline, FeaturePipeline,
                                  FeaturePipelineModel,
                                  MetadataFeaturePipeline, TfidfFeaturePipeline,
                                  PCAFeaturePipeline}

import scala.collection.JavaConversions._

case class Config(local_file: String = "", s3_bucket_file: String = "", gs_file: String = "", pipeline: String = "bag", limit: Int = -1, mode: String = "linear", reg: Double = 0.0, buckets: Int = 50000);

object RedditPrediction {

  def main(args: Array[String]) {

    val parser = new scopt.OptionParser[Config]("scopt") {
      head("predict reddit", "1.0")
      opt[String]('l', "local_file") action { (x, c) =>
        c.copy(local_file = x) } text("specify the full path of a local json file")
      opt[String]('a', "s3_file") action { (x, c) =>
        c.copy(s3_bucket_file = x) } text("subreddit name for s3 bucket access")
      opt[String]('g', "gs_file") action { (x, c) =>
        c.copy(gs_file = x) } text("subreddit name for s3 bucket access")
      opt[String]('m', "mode") action { (x, c) =>
        c.copy(mode = x) } text("""algorithm mode: [linear (regression), 
                                   |ridge (regression), logistic, 
                                   |forest (random forest regression)""".stripMargin)
      opt[String]('p', "pipeline") action { (x, c) =>
        c.copy(pipeline = x) } text("""features pipeline mode: 
                                       |[bag (of words), sentiment, 
                                       |metadata, tfidf, pca]""".stripMargin)
      opt[Int]('l', "limit") action { (x, c) =>
        c.copy(limit = x) } text("Limit comments to a certain amount")
      opt[Double]('r', "reg_param") action { (x, c) =>
        c.copy(reg = x) } text("regularization param for linear regression")
      opt[Int]('b', "buckets") action { (x, c) =>
        c.copy(buckets = x) } text("Number of buckets for the hashing trick, or vocab size for count vectorizing")
      help("help") text("prints this usage text")
    }

    val sc = new SparkContext()
    var df: DataFrame = null;

    var mode : String = ""
    var pipeline_mode: String = ""
    var limit: Int = -1;
    var reg_param: Double = 0.0;
    var buckets: Int = 50000;

    parser.parse(args, Config()) match { 
      case Some(config) => {

        mode = config.mode;
        limit = config.limit;
        pipeline_mode = config.pipeline
        reg_param = config.reg;
        buckets = config.buckets;

        // GCS BigQuery configuration
        if (config.gs_file.length > 0) { 

          val fullTableId = "cs260-1128:15_09." + config.gs_file;
          val projectId = "cs260-1128";
          val bucket = "15_09";

          val GSAccountEnableKey = "google.cloud.auth.service.account.enable";
          val GSKeyFileKey = "google.cloud.auth.service.account.keyfile";
          val GSEmailKey = "google.cloud.auth.service.account.email";

          val conf = sc.hadoopConfiguration;
          conf.set(BigQueryConfiguration.PROJECT_ID_KEY, projectId)
          conf.set(BigQueryConfiguration.GCS_BUCKET_KEY, bucket)
          BigQueryConfiguration.configureBigQueryInput(conf, fullTableId)
          conf.set("fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
          conf.set("fs.gs.project.id", projectId)
          conf.set("fs.gs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")

          conf.set(GSKeyFileKey, "/root/data/cs260-gs-key.p12")
          conf.set(GSEmailKey, "account-2@cs260-1128.iam.gserviceaccount.com")

          val columns = "body author created_utc subreddit_id link_id parent_id score retrieved_on controversiality gilded id subreddit ups".split(" ")
          val sqlContext = new SQLContext(sc);
          val schema =
            StructType(columns.map(fieldName => StructField(fieldName, StringType, true)))

          println(s"Loading from bigquery bucket: ${fullTableId}")
          val RDD = sc.newAPIHadoopRDD(
            conf, 
            classOf[GsonBigQueryInputFormat],
            classOf[LongWritable],
            classOf[JsonObject])

          val rowRDD = RDD.map({ pair: (LongWritable, JsonObject) => 
            Row.fromSeq(columns.map({ k: String => pair._2.get(k).getAsString }))
          })
          df = sqlContext.createDataFrame(rowRDD, schema).cache()
          println(s"Loaded ${df.count()} rows")

        } else if (config.s3_bucket_file.length > 0) {
          val conf = sc.hadoopConfiguration;
          conf.set("fs.s3.awsAccessKeyId", sys.env("AWS_ACCESS_KEY_ID"))
          conf.set("fs.s3.awsSecretAccessKey", sys.env("AWS_SECRET_ACCESS_KEY"))

          val sqlContext = new SQLContext(sc);
          val s3Path = "s3n://cs260-yoavz/" + config.s3_bucket_file + ".json"
          println(s"Loading from amazon s3 path: ${s3Path}")
          df = sqlContext.read.json(s3Path).cache

        } else if (config.local_file.length > 0) {
          val sqlContext = new SQLContext(sc);
          println(s"Loading from local path: ${config.local_file}")
          df = sqlContext.read.json(config.local_file);
        } else {
          println("Must specify a local or remote file!");
          return
        }
      }

      case None => {
        println("argument error")
        return
      }
    }

    println(s"${df.count()} total comments");

    // Filtering logic for removed and deleted comments
    val filtered = df.filter(df("author") !== "[deleted]")
                     .filter(df("body") !== "[removed]")
                     .filter(df("body") !== "[deleted]");
    println(s"${filtered.count()} total comments after filtering");

    // Limiting
    val limited = if (limit > 0) { 
      filtered.limit(limit).cache
    } else {
      filtered.cache
    }
    println(s"${limited.count()} comments after limiting");

    // Preprocessing
    var featurePipeline: FeaturePipeline = null;
    if (pipeline_mode == "sentiment") {
      println("Using sentiment feature pipeline")
      featurePipeline = new SentimentFeaturePipeline();
    } else if (pipeline_mode == "metadata") {
      println("Using metadata feature pipeline")
      featurePipeline = new MetadataFeaturePipeline();
    } else if (pipeline_mode == "tfidf") {
      println("Using tfidf feature pipeline")
      featurePipeline = new TfidfFeaturePipeline(buckets);
    } else if (pipeline_mode == "pca") {
      println(s"Using pca feature pipeline, using k = ${buckets}")
      featurePipeline = new PCAFeaturePipeline(buckets);
    } else {
      println("Using bag of words feature pipeline")
      featurePipeline = new FeaturePipeline(buckets);
    }

    val featurePipelineModel: FeaturePipelineModel = featurePipeline.fit(limited)
    val featured = featurePipelineModel.transform(limited) 
    println(s"${featured.count()} total comments after preprocessing");
    featured.select("body", "words", "features").take(10).foreach(println)

    // Split Train/Test
    val train_to_test = 0.70;
    val Array(train, test) = featured.randomSplit(Array(train_to_test, 1-train_to_test));
    println(s"Split into ${train.count()} training and ${test.count()} test comments");

    val regs: Array[Double] = (-5 to 5).toArray.map(x => scala.math.pow(2, x))
    // Do the training
    if (mode == "logistic") {
      println("Learning using Logistic Regression");
      val logistic = new RedditLogisticRegression();
      logistic.trainWithRegularization(train, regs)
      logistic.test(test)
      println("-") 
      println(s"Actual (training) distribution:")
      val bucketizer: CommentBucketizerModel = 
        featurePipelineModel.model.stages(4).asInstanceOf[CommentBucketizerModel]
      bucketizer.explainBucketDistribution(featured, "score_bucket")
    } else if (mode == "reg") {
      println("Learning using Linear Regression (reg search)");
      val regr = new RedditRidgeRegression();
      var history: Seq[(Double, Double, Double)] = regs.map{ reg =>
        val train_rmse = regr.train(train, reg)
        val test_rmse = regr.test(test)
        (reg, train_rmse, test_rmse)
      }
      history.foreach(println)
    } else if (mode == "ridge") {
      println("Learning using Ridge Regression");
      val regr = new RedditRidgeRegression();
      regr.train(train, regs);
      regr.test(test);
      featurePipelineModel.explainWeights(
        regr.getRegressionModel.weights.toArray, 10)
    } else if (mode == "forest") {
      println("Learning using Random Forest Regression");
      val regr = new RedditRandomForestRegression();
      regr.train(train);
      regr.test(test);
    } else if (mode == "sgd") {
      println("Learning using Linear Regression SGD")
      val regr = new RedditRegression();
      regr.trainSGD(train, test);
    } else {
      println(s"Learning using Vanilla Linear Regression, using reg param: ${reg_param}");
      val regr = new RedditRegression();
      regr.train(train, reg_param);
      regr.test(test);
      featurePipelineModel.explainWeights(
        regr.getRegressionModel.weights.toArray, 10)
      // regr.getRegressionModel.summary.objectiveHistory.zipWithIndex.foreach{ 
      //   case (hist, iter) => println(s"Iteration ${iter}: ${hist}")
      // }
    }
  }
}
