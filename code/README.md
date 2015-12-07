To Deploy an EC2 cluster 
========================
1. Start the EC2 cluster with the spark-ec2 script

2. SCP everything

    bin/export_files.sh "ec2 master goes here"

3. Login to master and rsync the data dir

    spark-ec2/copy-dir data

5. Run the spark-submit script on the jar

    spark/bin/spark-submit --master spark://ec2-54-219-54-139.us-west-1.compute.amazonaws.com:7077 reddit-prediction-assembly-1.0.jar -g compsci -m linear -u
