#!/bin/sh
#'java -cp ~/Drivers/RankLib/RankLib-2.8.jar ciir.umass.edu.features.FeatureManager -input train.csv -output mydata/ -k 5'
#java -Xmx15400m -jar ~/Drivers/RankLib/RankLib-2.8.jar -train train_less.csv -validate val_less.csv -ranker 6 -metric2t NDCG@4000 -save lambdaMart_v1.txt -shrinkage 0.05 -estop 170 -round 10000 -mls 5
#java -Xmx15400m -jar ~/Drivers/RankLib/RankLib-2.8.jar -load lambdaMart_v1.txt -rank test.csv -score predictions.csv

java -Xmx450g -jar RankLib-2.8.jar -train train.csv -validate val.csv -ranker 6 -metric2t NDCG@20000 -save lambdaMart_v1.txt -shrinkage 0.05 -estop 170 -round 10000 -mls 5
