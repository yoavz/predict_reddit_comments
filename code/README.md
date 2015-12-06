To Deploy an EC2 cluster 
========================
1. Start the EC2 cluster with the spark-ec2 script
2. Somehow get the comment json file into HDFS on the master node

    ephemeral-hdfs/bin/start-all.sh
    ephemeral-hdfs/bin/hadoop distcp s3n://cs260-yoavz/hiphopheads.json /root/hiphopheads.json

3. SCP the jar into the master node

    scp -i ~/keys/cs260.pem target/reddit-prediction-assembly-1.0.jar root@ec2-54-177-1-189.us-west-1.compute.amazonaws.com:

4. SCP the AFINN-111.txt into the master node

    scp -i ~/keys/cs260.pem /root/data/AFINN-111.txt root@ec2-54-177-1-189.us-west-1.compute.amazonaws.com:

5. Run the spark-submit script on the jar

    $SPARK_HOME/bin/spark-submit \
        --master spark://ec2-54-193-159-102.us-west-1.compute.amazonaws.com:6066 \
        target/reddit-prediction-assembly-1.0.jar $@
