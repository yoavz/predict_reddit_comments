sbt assembly && 
$SPARK_HOME/bin/spark-submit \
    --master spark://ec2-54-193-159-102.us-west-1.compute.amazonaws.com:6066 \
    target/reddit-prediction-assembly-1.0.jar $@
