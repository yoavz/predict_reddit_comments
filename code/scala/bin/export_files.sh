#!/bin/bash

BASE_DIR="$HOME/school/cs260/project/code/scala"
KEY_FILE="$HOME/keys/cs260.pem"
CRED_FILE="$HOME/keys/cs260-gs-key.p12"
AFINN_FILE="/root/data/AFINN-111.txt"
LOG_FILE="$SPARK_HOME/conf/log4j.properties"

MASTER="$@"

scp -i $KEY_FILE $BASE_DIR/target/reddit-prediction-assembly-1.0.jar root@$MASTER: 
scp -i $KEY_FILE $AFINN_FILE root@$MASTER:AFINN-111.txt 
scp -i $KEY_FILE $CRED_FILE root@$MASTER:cs260-gs-key.p12
scp -i $KEY_FILE $LOG_FILE root@$MASTER:spark/conf/log4j.properties

echo "Don't forget to rsync the data folder to all slaves on master!"
echo $MASTER
