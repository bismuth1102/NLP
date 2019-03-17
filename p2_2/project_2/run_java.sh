#!/bin/bash

java -cp bin/:jars/jama.jar HMM
java -cp bin/ Evaluator data/p1/test.txt results/p2/predictions.txt
