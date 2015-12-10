#!/bin/bash

# http://stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
BIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$BIN_DIR/.."
DATA_DIR="$BASE_DIR/data"
source $BIN_DIR/secrets.sh

# Constants
KEY_NAME="cs260"
KEY_FILE="$HOME/keys/cs260.pem"
NUM_SLAVES=1
REGION="us-west-1"
INSTANCE_TYPE="m1.xlarge"

OPTIONS="-k $KEY_NAME
 -i $KEY_FILE
 -r $REGION
 -s $NUM_SLAVES
 -t $INSTANCE_TYPE
 --copy-aws-credentials"

$SPARK_HOME/ec2/spark-ec2 $OPTIONS $@ 
