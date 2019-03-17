To run this project:
1. Run build_java.sh.
2. Run run_java.sh. This bash file runs HMM and Evaluator. There's two parameters for Evaluator: first is the groundTruth file, such as "/data/p1/test.txt"; second is the prediction file, such as "results/p1/predictions.txt". So you when you run "./run_java.sh", you need to add two parameters behind it. For example: "./run_java.sh /data/p1/test.txt results/p1/predictions.txt". This step will generate two files in results/p1, one is "log.txt" and the other is "predictions.txt". And the accuracy will be printed at terminal.
The thinking of HMM.java is pretty simple. First, initialize sets, maps labeled_corpus and unlabeled_corpus. Second, establish A, B and pi. Finally, run forward and Viterbi.
