#!/bin/bash

# try different Mu with data/p1/test.txt
for mu in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
do
	java -cp jars/jama.jar:bin/ HMM data/p1/train.txt data/p1/test.txt results/p3/predictions_${mu}.txt results/p3/log_${mu}.txt $mu
done

# evaluate the results
for mu in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
do
	java -cp bin/ Evaluator data/p1/test.txt results/p3/predictions_${mu}.txt
done

# try different Mu with data/p1/test.txt concatenated with data/p2/unlabeled_20news.txt
for mu in 0.3 0.5 0.7
do
	java -cp jars/jama.jar:bin/ HMM data/p1/train.txt data/p3/concatenated.txt results/p3/predictions_concatenated_${mu}.txt results/p3/log_concatenated_${mu}.txt $mu
done
