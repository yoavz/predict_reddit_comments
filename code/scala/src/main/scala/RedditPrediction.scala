package redditprediction

import scala.collection.JavaConversions._

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.NaiveBayesModel
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
                                  PCAFeaturePipeline, NaiveBayesFeaturePipeline}

import org.apache.log4j.Logger
import org.apache.log4j.Level

case class Config(local_file: String = "", s3_bucket_file: String = "", gs_file: String = "", pipeline: String = "bag", limit: Int = -1, mode: String = "linear", reg_param: Double = 0.0, buckets: Int = 50000);

object RedditPrediction {

  var config: Config = null;
  val log: Logger = Logger.getLogger("redditprediction")
  log.setLevel(Level.INFO)

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
        c.copy(reg_param = x) } text("regularization param for linear regression")
      opt[Int]('b', "buckets") action { (x, c) =>
        c.copy(buckets = x) } text("Number of buckets for the hashing trick, or vocab size for count vectorizing")
      help("help") text("prints this usage text")
    }

    val sc = new SparkContext()
    var df: DataFrame = null;

    parser.parse(args, Config()) match { 
      case Some(config) => {
        this.config = config;

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

          log.info(s"Loading from bigquery bucket: ${fullTableId}")
          val RDD = sc.newAPIHadoopRDD(
            conf, 
            classOf[GsonBigQueryInputFormat],
            classOf[LongWritable],
            classOf[JsonObject])

          val rowRDD = RDD.map({ pair: (LongWritable, JsonObject) => 
            Row.fromSeq(columns.map({ k: String => pair._2.get(k).getAsString }))
          })
          df = sqlContext.createDataFrame(rowRDD, schema).cache()
          log.info(s"Loaded ${df.count()} rows")

        } else if (config.s3_bucket_file.length > 0) {
          val conf = sc.hadoopConfiguration;
          conf.set("fs.s3.awsAccessKeyId", sys.env("AWS_ACCESS_KEY_ID"))
          conf.set("fs.s3.awsSecretAccessKey", sys.env("AWS_SECRET_ACCESS_KEY"))

          val sqlContext = new SQLContext(sc);
          val s3Path = "s3n://cs260-yoavz/" + config.s3_bucket_file + ".json"
          log.info(s"Loading from amazon s3 path: ${s3Path}")
          df = sqlContext.read.json(s3Path).cache

        } else if (config.local_file.length > 0) {
          val sqlContext = new SQLContext(sc);
          log.info(s"Loading from local path: ${config.local_file}")
          df = sqlContext.read.json(config.local_file);
        } else {
          log.info("Must specify a local or remote file!");
          return
        }
      }

      case None => {
        log.info("argument error")
        return
      }
    }

    log.info(s"${df.count()} total comments");

    // Filtering logic for removed and deleted comments
    val filtered = df.filter(df("author") !== "[deleted]")
                     .filter(df("body") !== "[removed]")
                     .filter(df("body") !== "[deleted]");
    log.info(s"${filtered.count()} total comments after filtering");

    // Limiting
    val limited = if (config.limit > 0) { 
      filtered.limit(config.limit).cache
    } else {
      filtered.cache
    }
    log.info(s"${limited.count()} comments after limiting");

    // Preprocessing
    var featurePipeline: FeaturePipeline = null;
    if (config.mode == "binary")  {
      log.info("Using binary bayes feature pipeline")
      featurePipeline = new NaiveBayesFeaturePipeline(true);
    } else if (config.mode == "bayes") {
      // bayes mode overrides all other pipeline modes
      log.info("Using bayes feature pipeline")
      featurePipeline = new NaiveBayesFeaturePipeline();
    } else if (config.pipeline == "sentiment") {
      log.info("Using sentiment feature pipeline")
      featurePipeline = new SentimentFeaturePipeline();
    } else if (config.pipeline == "metadata") {
      log.info("Using metadata feature pipeline")
      featurePipeline = new MetadataFeaturePipeline();
    } else if (config.pipeline == "tfidf") {
      log.info("Using tfidf feature pipeline")
      featurePipeline = new TfidfFeaturePipeline(config.buckets);
    } else if (config.pipeline == "pca") {
      log.info(s"Using pca feature pipeline, using k = ${config.buckets}")
      featurePipeline = new PCAFeaturePipeline(config.buckets);
    } else if (config.pipeline == "all")  {
      val pipelines: List[(String, FeaturePipeline)] = 
        List(("bag", new FeaturePipeline(config.buckets)), 
             ("tfidf", new TfidfFeaturePipeline(config.buckets)),
             ("metadata", new MetadataFeaturePipeline()))
      pipelines.map(t =>  (t._1, trainWithPipeline(t._2, limited))).foreach(log.info)
      return;
    } else if (config.pipeline == "bin_class") { 
      config = config.copy(mode = "binary")
      val bayes_acc = trainWithPipeline(new NaiveBayesFeaturePipeline(true), limited)
      config = config.copy(mode = "logistic")
      val log_acc = trainWithPipeline(new NaiveBayesFeaturePipeline(true), limited)
      log.info(s"Naive Bayes Test Accuracy: ${bayes_acc}")
      log.info(s"Logistic Test Accuracy: ${log_acc}")
      return
    } else if (config.pipeline == "multi_class") {
      config = config.copy(mode = "bayes")
      val bayes_acc = trainWithPipeline(new NaiveBayesFeaturePipeline(false), limited)
      config = config.copy(mode = "logistic")
      val log_acc = trainWithPipeline(new FeaturePipeline(config.buckets), limited)
      log.info(s"Naive Bayes Test Accuracy: ${bayes_acc}")
      log.info(s"Logistic Test Accuracy: ${log_acc}")
      return
    } else {
      log.info("Using bag of words feature pipeline")
      featurePipeline = new FeaturePipeline(config.buckets);
    }

    trainWithPipeline(featurePipeline, limited)
  }

  def trainWithPipeline(featurePipeline: FeaturePipeline, dataset: DataFrame): Double = {

    val featurePipelineModel: FeaturePipelineModel = featurePipeline.fit(dataset)
    val featured = featurePipelineModel.transform(dataset) 
    // featured.select("body", "words", "features").take(10).foreach(log.info)

    // Split Train/Test
    val train_to_test = 0.70;
    val random_seed = 11L;

    val Array(train, test) = featured.randomSplit(Array(train_to_test, 1-train_to_test), random_seed);
    log.info(s"Split into ${train.count()} training and ${test.count()} test comments");

    val regs: Array[Double] = (-5 to 5).toArray.map(x => scala.math.pow(2, x))
    // Do the training
    if (config.mode == "binary") {
      log.info("Learning using Binary Naive Bayes");
      val bayes = new RedditNaiveBayes(true);
      bayes.train(train)
      val test_accu: Double = bayes.test(test)
    
      val bucketizer: CommentBucketizerModel = 
        featurePipelineModel.getCommentBucketizerModel
      if (bucketizer != null) {
        log.info(s"Actual (training) distribution:")
        bucketizer.explainBucketDistribution(featured, "score_bucket", log)
      }

      val nbModel: NaiveBayesModel =
        bayes.getModel.stages(0).asInstanceOf[NaiveBayesModel]
      featurePipelineModel.explainWeights(
        getWeightsNB(nbModel).toArray, 10)
      log.info(s"Test accuracy: ${test_accu}")
      return test_accu
    } else if (config.mode == "bayes") {
      log.info("Learning using Naive Bayes");
      val bayes = new RedditNaiveBayes(false);
      bayes.train(train)
      val test_accu: Double = bayes.test(test)
    
      val bucketizer: CommentBucketizerModel = 
        featurePipelineModel.getCommentBucketizerModel
      if (bucketizer != null) {
        log.info(s"Actual (training) distribution:")
        bucketizer.explainBucketDistribution(featured, "score_bucket", log)
      }

      val nbModel: NaiveBayesModel =
        bayes.getModel.stages(0).asInstanceOf[NaiveBayesModel]
      featurePipelineModel.explainWeights(
        getWeightsNB(nbModel).toArray, 10)
      log.info(s"Test accuracy: ${test_accu}")
      return test_accu
    } else if (config.mode == "logistic") {
      log.info("Learning using Logistic Regression");
      val logistic = new RedditLogisticRegression();
      logistic.trainWithRegularization(train, regs)
      val test_accu: Double = logistic.test(test)

      val bucketizer: CommentBucketizerModel = 
        featurePipelineModel.getCommentBucketizerModel
      if (bucketizer != null) {
        log.info(s"Actual (training) distribution:")
        bucketizer.explainBucketDistribution(featured, "score_bucket", log)
      }

      log.info(s"Test accuracy: ${test_accu}")
      return test_accu
    } else if (config.mode == "reg") {
      log.info("Learning using Linear Regression (reg search)");
      val regr = new RedditRidgeRegression();
      var history: Seq[(Double, Double, Double)] = regs.map{ reg =>
        val train_rmse = regr.train(train, reg);
        val test_rmse = regr.test(test)
        (reg, train_rmse, test_rmse)
      }
      history.foreach(log.info)
      return 0.0 // can ignore the return value of this
    } else if (config.mode == "ridge") {
      log.info("Learning using Ridge Regression");
      val regr = new RedditRidgeRegression();
      val (train_best_reg, train_rmse) = regr.train(train, regs);
      log.info(s"Training RMSE: ${train_rmse}, Best Lambda: ${train_best_reg}");
      featurePipelineModel.explainWeights(
        regr.getRegressionModel.weights.toArray, 10)
      val test_rmse: Double = regr.test(test) 
      log.info(s"Testing RMSE: ${test_rmse}")
      return test_rmse
    } else if (config.mode == "forest") {
      // TODO: print if you use these vals
      log.info("Learning using Random Forest Regression");
      val regr = new RedditRandomForestRegression();
      regr.train(train);
      regr.test(test);
    } else if (config.mode == "sgd") {
      log.info("Learning using Linear Regression SGD")
      val regr = new RedditRegression();
      regr.trainSGD(train, test);
    } else {
      log.info(s"Learning using Vanilla Linear Regression, using reg param: ${config.reg_param}");
      val regr = new RedditRegression();
      val train_rmse = regr.train(train, config.reg_param);
      log.info(s"Training RMSE: ${train_rmse}")
      featurePipelineModel.explainWeights(
        regr.getRegressionModel.weights.toArray, 10)
      val test_rmse = regr.test(test);
      log.info(s"Testing RMSE: ${test_rmse}")
      // regr.getRegressionModel.summary.objectiveHistory.zipWithIndex.foreach{ 
      //   case (hist, iter) => log.info(s"Iteration ${iter}: ${hist}")
      // }
    }
    0.0
  }

  def getWeightsNB(model: NaiveBayesModel) = {
    (0 to model.theta.numCols).map{ t =>
      model.theta.apply(0, t)
    }
  }
}
