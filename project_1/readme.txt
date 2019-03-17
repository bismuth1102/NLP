I put 3 files in the project: ProbEstimator.java, Predictor.java and pom.xml. To run the project on sunlab, the steps are as following:
a. ./build_java.sh
b. ./run_java.sh

Things I have done to optimize run time:
a. I use a HashMap to map bigrams and its coordinate in the matrix. I only have O(Nlog(n)) when establishing the matrix to calculate c by this method. (N means total bigrams in trainning set, n means distinctive bigrams in trainning set). Otherwise, I need to compare string for each bigram when locating its coordinate and it will give me O(Nn).
b. I use arraylists instead of linkedlists. Because arraylist will hit the target directly when using arraylist.get(i), which is O(1); and linkedlist will give me O(n).
c. When I need to write to a file, I joint the results and write one string to the file at last for those functions which have not too many results in order to reduce io time. On the other way, I write to the file after jointing some of the results, not all of them, when there are so many results. Because string joint consume lots of memory.