#!/bin/bash

source bin/secrets.sh && 
sbt assembly && 
$SPARK_HOME/bin/spark-submit target/reddit-prediction-assembly-1.0.jar $@
