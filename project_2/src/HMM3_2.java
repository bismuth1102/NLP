import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;

import Jama.Matrix;

public class HMM3_2 {
	HashSet<String> uniqueUnigramSet = new HashSet<String>();	//hashset for remove duplicate
	HashSet<String> uniqueTagSet = new HashSet<String>();
	Map<String, Integer> vocabulary = new HashMap<String, Integer>();
	Map<String, Integer> pos_tags = new HashMap<String, Integer>();
	Map<Integer, String> inv_pos_tags =  new HashMap<Integer, String>();
	Matrix mleA;
	Matrix mleB;
	Matrix mlepi;
	Matrix A;
	Matrix B;
	Matrix pi;
	Matrix alpha;
	Matrix beta;
	ArrayList<Sentence> labeled_corpus = new ArrayList<Sentence>();
	ArrayList<Sentence> unlabeled_corpus = new ArrayList<Sentence>();

	double mu = 0.3;
	int num_words = 0;
	int num_postags = 0;
	int test_sentences = 0;
	int max_sentence_length = 0;
	double smoothing_eps = 0.00001;
	ArrayList<Matrix> alphas = new ArrayList<Matrix>();	//store all sentences'alphas in one em'iteration
	ArrayList<Matrix> betas = new ArrayList<Matrix>();	//store all sentences'betas in one em'iteration
	
	public static void main(String[] args) {
		HMM3_2 hmm = new HMM3_2();
		
		hmm.init();
		
		//mle
		hmm.fillMap();
		hmm.prepareMatrices();
		hmm.maxSentenceLength();
		
		hmm.em();
		
	}
	
	//initialize labeled_corpus and unlabeled_corpus
	private void init() {
		FileHandler fh = new FileHandler();
		try {
			labeled_corpus = fh.readTaggedSentences("data/p1/train.txt");
			unlabeled_corpus = fh.readTaggedSentences("data/p2/concatenated.txt");
			test_sentences = unlabeled_corpus.size();
		} catch (IOException e) {
			e.printStackTrace();
		}
		addUniques("data/p1/train.txt");
		addUniques("data/p2/concatenated.txt");
	}
	
	//get unique words and tags in train.txt
	private void addUniques(String filename) {
		ArrayList<String> words = new ArrayList<String>();
		words = read(filename);
		for (int i = 0; i < words.size(); i++) {
			String tempString = words.get(i); 
			if (tempString.length() != 0) {
				String[] parts = tempString.split(" ");
				uniqueUnigramSet.add(parts[0]);
				uniqueTagSet.add(parts[1]);
			}
		}
	}
	
	private void fillMap() {
		Iterator<String> unigram_it = uniqueUnigramSet.iterator();
		for(int i = 0; i < uniqueUnigramSet.size(); i++) {
			String unigram = unigram_it.next();
			vocabulary.put(unigram, i);
		}
		
		Iterator<String> tag_it = uniqueTagSet.iterator();
		for(int i = 0; i < uniqueTagSet.size(); i++) {
			String tag = tag_it.next();
			pos_tags.put(tag, i);
			inv_pos_tags.put(i, tag);
		}
	}
	
	private void prepareMatrices() {
		num_words = vocabulary.size();
		num_postags = pos_tags.size();
		
		mleA = new Matrix(num_postags, num_postags, smoothing_eps);
		double[] ARowSum = new double[num_postags];
		mleB = new Matrix(num_postags, num_words, smoothing_eps);
		double[] BRowSum = new double[num_postags];
		mlepi = new Matrix(1, num_postags, smoothing_eps);
		
		//A
		for (int i = 0; i < labeled_corpus.size(); i++) {
			Sentence s = labeled_corpus.get(i);
			for (int k = 0; k < s.length()-1; k++) {
				Word preWord = s.getWordAt(k);
				Word postWord = s.getWordAt(k+1);
				int preTagNum = pos_tags.get(preWord.getPosTag());
				int postTagNum = pos_tags.get(postWord.getPosTag());
				double Aij = mleA.get(preTagNum, postTagNum);
				Aij++;
				mleA.set(preTagNum, postTagNum, Aij);
				ARowSum[preTagNum]++;
			}
		}
		
		//B
		for (int i = 0; i < labeled_corpus.size(); i++) {
			Sentence s = labeled_corpus.get(i);
			
			ArrayList<Integer> indexArray = vocabularyIndexArray(s);
			
			for (int k = 0; k < s.length(); k++) {
				Word w = s.getWordAt(k);
				int tagNum = pos_tags.get(w.getPosTag());
				int wordNum = indexArray.get(k);
				double Bij = mleB.get(tagNum, wordNum);
				Bij++;
				mleB.set(tagNum, wordNum, Bij);
				BRowSum[tagNum]++;
			}
		}
		
		//pi
		for (int i = 0; i < labeled_corpus.size(); i++) {
			Sentence s = labeled_corpus.get(i);
			Word begin = s.getWordAt(0);
			int tagNum = pos_tags.get(begin.getPosTag());
			double picount = mlepi.get(0, tagNum);
			picount++;
			mlepi.set(0, tagNum, picount);
		}

		//normalize
		//A
		for (int i = 0; i < num_postags; i++) {
			double sumRow = ARowSum[i];
			for (int j = 0; j < num_postags; j++) {
				double count = mleA.get(i, j);
				mleA.set(i, j, count/sumRow);
			}
		}
		A = mleA.copy();
		
		//B
		for (int i = 0; i < num_postags; i++) {
			double sumRow = BRowSum[i];
			for (int j = 0; j < num_words; j++) {
				double count = mleB.get(i, j);
				mleB.set(i, j, count/sumRow);
			}
		}
		B = mleB.copy();
		
		//pi
		double sumRow = labeled_corpus.size();
		for (int j = 0; j < num_postags; j++) {
			double count = mlepi.get(0, j);
			mlepi.set(0, j, count/sumRow);
		}
		pi = mlepi.copy();
	}
	
	private void maxSentenceLength() {
		for (int s = 0; s < test_sentences; s++) {
			Sentence sentence = unlabeled_corpus.get(s);
			if(max_sentence_length < sentence.length()) {
				max_sentence_length = sentence.length();
			}
		}
	}
	
	private ArrayList<Integer> vocabularyIndexArray(Sentence sentence){
		ArrayList<Integer> array = new ArrayList<Integer>();
		
		for(Word word : sentence) {
			String lemme = word.getLemme();
			int i = vocabulary.get(lemme);
			array.add(i);
		}
		
		return array;
	}
	
	private double forward(Sentence sentence) {
		
		ArrayList<Integer> indexArray = vocabularyIndexArray(sentence);
		double[] c = new double[sentence.length()];
		
		int lemmeNum = indexArray.get(0);
		double sum = 0;	//use once every column
		double lastSum = 1;
		
		for (int i = 0; i < num_postags; i++) {	//pi
			double prob = pi.get(0, i)*B.get(i, lemmeNum);
			alpha.set(i, 0, prob);
			sum += prob;
		}
		c[0] = 1.0/sum;
		for(int i = 0; i < num_postags; i++) {	//normalize
			double prob = alpha.get(i, 0);
			prob *= c[0];
			alpha.set(i, 0, prob);
		}
		
		for (int j = 1; j < sentence.length(); j++) {
			sum = 0;
			for (int i = 0; i < num_postags; i++) {	//rows of column j
				double prob = 0;
				for (int k = 0; k < num_postags; k++) {	//rows of column j-1
					prob += alpha.get(k, j-1)*A.get(k, i);
				}
				prob *= B.get(i, indexArray.get(j));
				alpha.set(i, j, prob);
				sum += prob;
			}
			c[j] = 1.0/sum;
			for(int i = 0; i < num_postags; i++) {	//normalize
				double prob = alpha.get(i, j);
				prob *= c[j];
				alpha.set(i, j, prob);
			}
		}
		
		for(int i = 0; i < sentence.length(); i++) {
			lastSum *= c[i];
			if (lastSum>1.0E300) {	//in case c1*c2*...*cT is bigger than what Java can implement 
				lastSum = 1.0E299;
				break;
			}
		}
		lastSum = 1.0/lastSum;
		return Math.log(lastSum);
	}
	
	private void backward(Sentence sentence) {
		
		ArrayList<Integer> indexArray = vocabularyIndexArray(sentence);
		double[] c = new double[sentence.length()];
		
		double sum = 0;	//use once every column
		
		for (int i = 0; i < num_postags; i++) {	//last column
			beta.set(i, sentence.length()-1, 1.0/num_postags);
		}
		c[0] = 1.0/num_postags;
		
		for (int j = sentence.length()-2; j >= 0; j--) {
			sum = 0;
			for (int i = 0; i < num_postags; i++) {	//rows of column j
				double prob = 0;
				for (int k = 0; k < num_postags; k++) {	//rows of column j+1
					prob += beta.get(k, j+1)*A.get(i, k)*B.get(k, indexArray.get(j+1));
				}
				beta.set(i, j, prob);
				sum += prob;
			}
			
			c[j] = 1.0/sum;
			for(int i = 0; i < num_postags; i++) {	//normalize
				double prob = beta.get(i, j);
				prob *= c[j];
				beta.set(i, j, prob);
			}
		}
			
	}
	
	private void viterbi() {
		try {
			File f = new File("results/p3/predictions_concatenated.txt");
			BufferedWriter out = new BufferedWriter(new FileWriter(f));
			
			for (int s = 0; s < test_sentences; s++) {	//all sentences
				Sentence sentence = unlabeled_corpus.get(s);
				
				ArrayList<Integer> indexArray = vocabularyIndexArray(sentence);
				
				Matrix logMatrix = new Matrix(num_postags, sentence.length(), 0);
				Matrix backMatrix = new Matrix(num_postags, sentence.length(), 0);
				int lemmeNum = indexArray.get(0);
				
				for (int i = 0; i < num_postags; i++) {	//pi
					double prob = Math.log(pi.get(0, i)) + Math.log(B.get(i, lemmeNum));
					logMatrix.set(i, 0, prob);
					backMatrix.set(i, 0, 0);
				}
				
				for (int j = 1; j < sentence.length(); j++) {
					for (int i = 0; i < num_postags; i++) {	//rows of column j
						double max = -99999;
						int back = 0;
						for (int k = 0; k < num_postags; k++) {	//rows of column j-1
							double prob = logMatrix.get(k, j-1) + 	//numerical issues
									Math.log(A.get(k, i)) + 
									Math.log(B.get(i, indexArray.get(j)));
							if (prob>max) {
								max = prob;
								back = k;
							}
						}
						logMatrix.set(i, j, max);
						backMatrix.set(i, j, back);
					}	//end of one column
				}	//end of one sentence 
				double max = -99999;
				double back = 0;
				for (int m = 0; m < num_postags; m++) {	//get biggest prob at the last column
					double prob = logMatrix.get(m, sentence.length()-1);
					if (prob > max) {
						max = prob;
						back = m;
					}
				}
				ArrayList<Double> backState = new ArrayList<Double>();
				backState.add(back);
				for (int m = sentence.length()-1; m > 0; m--) {
					back = backMatrix.get((int) back, m);
					backState.add(back);
				}
				Collections.reverse(backState);
				for (int m = 0; m < sentence.length(); m++) {
					Word w = sentence.getWordAt(m);
					String wlemme = w.getLemme();
					double realBack = backState.get(m);
					String predict = inv_pos_tags.get((int)realBack);
					String str = wlemme+" "+predict+"\n";
					
					out.write(str);
				}
				out.write("\n");
			}	//end of all sentences
		out.flush();
		out.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("prediction finished");
	}
	
	private void em() {
		
		ArrayList<Double> log_likelihoods = new ArrayList<Double>();
		
		for(int time = 0; time < 30; time++) {
			
			Matrix aUp = new Matrix(num_postags, num_postags, 0);	//numerator of aij
			double[] aDown = new double[num_postags];	//denominator of aij
			Matrix bUp = new Matrix(num_postags, num_words, 0);	//numerator of bk(j)
			double[] bDown = new double[num_postags];	//denominator of bk(j)
			Matrix sentencePi = new Matrix(1, num_postags, 0);
			double log_likelihood = 0;
			
			for (int s = 0; s < test_sentences; s++) {	//all sentences
				
				Sentence sentence = unlabeled_corpus.get(s);
				ArrayList<Integer> indexArray = vocabularyIndexArray(sentence);
				
				Matrix sumT_xi = new Matrix(num_postags, num_postags, 0);
				Matrix gamma = new Matrix(num_postags, sentence.length(), 0);
				alpha = new Matrix(num_postags, max_sentence_length, 0);
				beta = new Matrix(num_postags, max_sentence_length, 0);
				
				log_likelihood += forward(sentence);	//per iteration per sentence
				backward(sentence);
				
				//e
				for (int i = 0; i < num_postags; i++) {
					for (int j = 0; j < num_postags; j++) {
						double sumT = 0;
						for(int t = 0; t < sentence.length()-1; t++) {
							double prob = alpha.get(i, t)*beta.get(j, t+1)*A.get(i, j)*B.get(j, indexArray.get(t+1));
							sumT += prob;
						}
						sumT_xi.set(i, j, sumT);
					}
				}
				
				for (int i = 0; i < num_postags; i++) {
					for(int t = 0; t < sentence.length(); t++) {
						double prob = alpha.get(i, t)*beta.get(i, t);
						gamma.set(i, t, prob);
					}
				}
				
				//m
				for (int i = 0; i < num_postags; i++) {
					double aDownSum = 0;
					for (int j = 0; j < num_postags; j++) {
						aDownSum += sumT_xi.get(i, j);
						aUp.set(i, j, aUp.get(i, j)+sumT_xi.get(i, j));
					}
					aDown[i] += aDownSum;
				}
				
				for (int i = 0; i < num_postags; i++) {
					double bDownSum = 0;
					for(int t = 0; t < sentence.length(); t++) {
						int index = indexArray.get(t);
						bUp.set(i, index, bUp.get(i, index)+gamma.get(i, t));
						
						bDownSum += gamma.get(i, t);
					}
					bDown[i] += bDownSum;
				}
				
				for(int i = 0; i < num_postags; i++) {
					sentencePi.set(0, i, sentencePi.get(0, i)+gamma.get(i, 0));
				}
				
			}	//end all sentences
			
			//output log_likelihood
			System.out.println(log_likelihood);
			log_likelihoods.add(log_likelihood);
			
			//A=aUp/aDown, B=bUp/bDown, pi=pi[i]/sentence_number
			//add 1.0E-150 in case of out of number boundary(<1.0E-300)
			for(int i = 0; i < num_postags; i++) {
				for(int j = 0; j < num_postags; j++) {
					double prob = aUp.get(i, j)/aDown[i];
					A.set(i, j, prob+1.0E-150);
				}
				for(int t = 0; t < num_words; t++) {
					double prob = bUp.get(i, t)/bDown[i];
					B.set(i, t, prob+1.0E-150);
				}
				pi.set(0, i, sentencePi.get(0, i)/test_sentences+1.0E-150);
			}
			
			//add mu
			for(int i = 0; i < num_postags; i++) {
				for(int j = 0; j < num_postags; j++) {
					double _mle = mleA.get(i, j);
					double _em = A.get(i, j);
					A.set(i, j, _mle*mu+_em*(1-mu));
				}
				for(int t = 0; t < num_words; t++) {
					double _mle = mleB.get(i, t);
					double _em = mleB.get(i, t);
					B.set(i, t, _mle*mu+_em*(1-mu));
				}
				pi.set(0, i, mlepi.get(0, i)*mu+pi.get(0, i)*(1-mu));
			}
			
		}	//end all 30 times
		
		viterbi();
		
		try {
			BufferedWriter out = new BufferedWriter(new OutputStreamWriter(   
	                  new FileOutputStream("results/p3/log_concatenated.txt", true)));
			for(double i : log_likelihoods) {
				out.write(i+"\n");
				out.flush();
				out.close();
			}
		} catch (Exception e) {
			// TODO: handle exception
		}
	}
	
	private ArrayList<String> read(String fileName) {
        File file = new File(fileName);
        ArrayList<String> words = new ArrayList<String>();
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(file));
            String tempString = null;
            while ((tempString = reader.readLine()) != null) {
            	words.add(tempString);
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return words;
    }
	
}
