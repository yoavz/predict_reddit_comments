#!/bin/bash

sbt package
$SPARK_HOME/bin/spark-submit target/scala-2.10/predict-reddit-comments_2.10-1.0.jar $@
