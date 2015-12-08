#!/bin/bash

KEY_FILE="$HOME/keys/cs260.pem"
CRED_FILE="$HOME/keys/cs260-gs-key.p12"
AFINN_FILE="/root/data/AFINN-111.txt"

MASTER="$@"

scp -i $KEY_FILE $BASE_DIR/target/reddit-prediction-assembly-1.0.jar root@$MASTER: 
scp -i $KEY_FILE $AFINN_FILE root@$MASTER:AFINN-111.txt 
scp -i $KEY_FILE $CRED_FILE root@$MASTER:cs260-gs-key.p12

echo "Don't forget to rsync the data folder to all slaves on master!"
echo $MASTER
