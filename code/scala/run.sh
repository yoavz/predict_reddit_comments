#!/bin/bash

sbt package && 
$SPARK_HOME/bin/spark-submit target/predict-reddit-comments-1.0.jar $@
