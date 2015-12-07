#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
DATA_DIR="$BASE_DIR/data"

KEY_FILE="$HOME/keys/cs260.pem"
CRED_FILE="$HOME/keys/cs260-gs-key.p12"
AFINN_FILE="$DATA_DIR/AFINN-111.txt"

MASTER="$@"

scp -i $KEY_FILE $BASE_DIR/target/reddit-prediction-assembly-1.0.jar root@$MASTER: 
scp -i $KEY_FILE $AFINN_FILE root@$MASTER:/root/data/AFINN-111.txt 
scp -i $KEY_FILE $CRED_FILE root@$MASTER:/root/data/cs260-gs-key.p12

echo "Don't forget to rsync the data folder to all slaves on master!"
echo $MASTER