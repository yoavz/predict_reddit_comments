#!/bin/bash

declare -a arr=("hiphopheads" "videos" "programming" "television" "movies" "compsci" "AskReddit" "askscience" "nosleep" "music")

for i in "${arr[@]}"
do
    bq query --destination_table 15_09.$i "SELECT * FROM [fh-bigquery:reddit_comments.2015_09] WHERE subreddit == \"$i\";"
    echo "queried $i"
done
