package nlp;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;

import Jama.Matrix;

public class myHMM {
	HashSet<String> uniqueUnigramSet = new HashSet<String>();	//hashset for remove duplicate
	HashSet<String> uniqueTagSet = new HashSet<String>();
	Map<String, Integer> vocabulary = new HashMap<String, Integer>();
	Map<String, Integer> pos_tags = new HashMap<String, Integer>();
	Map<Integer, String> inv_pos_tags =  new HashMap<Integer, String>();
	Matrix A;
	Matrix B;
	Matrix pi;
	ArrayList<Sentence> labeled_corpus = new ArrayList<Sentence>();
	ArrayList<Sentence> unlabeled_corpus = new ArrayList<Sentence>();

	int num_words = 0;
	int num_postags = 0;
	double smoothing_eps = 0.1;
	ArrayList<Matrix> alphas = new ArrayList<Matrix>();	//store all sentences'alphas in one em'iteration
	ArrayList<Matrix> betas = new ArrayList<Matrix>();	//store all sentences'betas in one em'iteration
	
	public static void main(String[] args) {
		myHMM hmm = new myHMM();
		
		hmm.init();
		
		//mle
		hmm.fillMap();
		hmm.prepareMatrices();
		
		hmm.forward();
		hmm.backward();
		hmm.em();
		
	}
	
	//initialize labeled_corpus and unlabeled_corpus
	private void init() {
		FileHandler fh = new FileHandler();
		try {
			labeled_corpus = fh.readTaggedSentences("data/p1/train.txt");
			unlabeled_corpus = fh.readTaggedSentences("data/p1/test.txt");
		} catch (IOException e) {
			e.printStackTrace();
		}
		getTrainUnique("data/p1/train.txt");
		addTestUnique("data/p1/test.txt");
	}
	
	//get unique words and tags in train.txt
	private void getTrainUnique(String filename) {
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
	
	//get unique words and tags in test.txt
	private void addTestUnique(String filename) {
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
		
		A = new Matrix(num_postags, num_postags, smoothing_eps);
		double[] ARowSum = new double[num_postags];
		B = new Matrix(num_postags, num_words, smoothing_eps);
		double[] BRowSum = new double[num_postags];
		pi = new Matrix(1, num_postags, smoothing_eps);
		
		//A
		for (int i = 0; i < labeled_corpus.size(); i++) {
			Sentence s = labeled_corpus.get(i);
			for (int k = 0; k < s.length()-1; k++) {
				Word preWord = s.getWordAt(k);
				Word postWord = s.getWordAt(k+1);
				int preTagNum = pos_tags.get(preWord.getPosTag());
				int postTagNum = pos_tags.get(postWord.getPosTag());
				double Aij = A.get(preTagNum, postTagNum);
				Aij++;
				A.set(preTagNum, postTagNum, Aij);
				ARowSum[preTagNum]++;
			}
		}
		
		//B
		for (int i = 0; i < labeled_corpus.size(); i++) {
			Sentence s = labeled_corpus.get(i);
			for (int k = 0; k < s.length(); k++) {
				Word w = s.getWordAt(k);
				int tagNum = pos_tags.get(w.getPosTag());
				int wordNum = vocabulary.get(w.getLemme());
				double Bij = B.get(tagNum, wordNum);
				Bij++;
				B.set(tagNum, wordNum, Bij);
				BRowSum[tagNum]++;
			}
		}
		
		//pi
		for (int i = 0; i < labeled_corpus.size(); i++) {
			Sentence s = labeled_corpus.get(i);
			Word begin = s.getWordAt(0);
			int tagNum = pos_tags.get(begin.getPosTag());
			double picount = pi.get(0, tagNum);
			picount++;
			pi.set(0, tagNum, picount);
		}

		//normalize
		//A
		for (int i = 0; i < num_postags; i++) {
			double sumRow = ARowSum[i];
			for (int j = 0; j < num_postags; j++) {
				double count = A.get(i, j);
				A.set(i, j, count/sumRow);
			}
		}
		
		//B
		for (int i = 0; i < num_postags; i++) {
			double sumRow = BRowSum[i];
			for (int j = 0; j < num_words; j++) {
				double count = B.get(i, j);
				B.set(i, j, count/sumRow);
			}
		}
		
		//pi
		double sumRow = labeled_corpus.size();
		for (int j = 0; j < num_postags; j++) {
			double count = pi.get(0, j);
			pi.set(0, j, count/sumRow);
		}
	}
	
	private void forward() {
		
		alphas.clear();
		String lemme = "";
		double totalSum = 0;
		
		try {
			BufferedWriter out = new BufferedWriter(new OutputStreamWriter(   
	                  new FileOutputStream("results/p2/log.txt", true)));
			
			for (int s = 0; s < unlabeled_corpus.size(); s++) {	//all sentences
				
				Sentence sentence = unlabeled_corpus.get(s);
				Matrix alpha = new Matrix(num_postags, sentence.length(), 0);
				Matrix c = new Matrix(1, sentence.length(), 0);
				
				Word begin = sentence.getWordAt(0);
				lemme = begin.getLemme();
				int lemmeNum = vocabulary.get(lemme);
				double sum = 0;	//use once every column
				double lastSum = 1;
				
				for (int i = 0; i < num_postags; i++) {	//pi
					double prob = pi.get(0, i)*B.get(i, lemmeNum);
					alpha.set(i, 0, prob);
					sum += prob;
				}
				c.set(0, 0, 1.0/sum);
				for(int i = 0; i < num_postags; i++) {	//normalize
					double prob = alpha.get(i, 0);
					prob *= c.get(0, 0);
					alpha.set(i, 0, prob);
				}
				
				for (int j = 1; j < sentence.length(); j++) {
					Word w = sentence.getWordAt(j);
					lemme = w.getLemme();
					sum = 0;
					for (int i = 0; i < num_postags; i++) {	//rows of column j
						double prob = 0;
						for (int k = 0; k < num_postags; k++) {	//rows of column j-1
							prob += alpha.get(k, j-1)*A.get(k, i);
						}
						prob *= B.get(i, vocabulary.get(lemme));
						alpha.set(i, j, prob);
						sum += prob;
					}
					c.set(0, j, 1.0/sum);
					for(int i = 0; i < num_postags; i++) {	//normalize
						double prob = alpha.get(i, j);
						prob *= c.get(0, j);
						alpha.set(i, j, prob);
					}
				}	//end for one sentence
				
				for(int i = 0; i < sentence.length(); i++) {
					lastSum *= c.get(0, i);
					if (lastSum>1.0E300) {	//in case c1*c2*...*cT is bigger than what Java can implement 
						lastSum = 1.0E299;
						break;
					}
				}
				lastSum = 1.0/lastSum;
				totalSum += Math.log(lastSum);
				
				alphas.add(alpha);
			}	//end for all sentences
			out.write(totalSum+"\n");
			out.flush();
			out.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private void backward() {
		
		betas.clear();
		String lemme = "";
		
		for (int s = 0; s < unlabeled_corpus.size(); s++) {	//all sentences
			Sentence sentence = unlabeled_corpus.get(s);
			Matrix beta = new Matrix(num_postags, sentence.length(), 0);
			Matrix c = new Matrix(1, sentence.length(), 0);
			
			double sum = 0;	//use once every column
			
			for (int i = 0; i < num_postags; i++) {	//last column
				beta.set(i, sentence.length()-1, 1.0/num_postags);
			}
			c.set(0, 0, 1.0/num_postags);
			
			for (int j = sentence.length()-2; j >= 0; j--) {
				Word w = sentence.getWordAt(j+1);
				lemme = w.getLemme();
				sum = 0;
				for (int i = 0; i < num_postags; i++) {	//rows of column j
					double prob = 0;
					for (int k = 0; k < num_postags; k++) {	//rows of column j+1
						prob += beta.get(k, j+1)*A.get(i, k)*B.get(k, vocabulary.get(lemme));
					}
					beta.set(i, j, prob);
					sum += prob;
				}
				
				c.set(0, j, 1.0/sum);
				for(int i = 0; i < num_postags; i++) {	//normalize
					double prob = beta.get(i, j);
					prob *= c.get(0, j);
					beta.set(i, j, prob);
				}
			}	//end for one sentence
			
			betas.add(beta);
		}	//end for all sentences
		
	}
	
	private void viterbi() {
		String lemme = "";
		try {
			File f = new File("results/p2/predictions.txt");
			BufferedWriter out = new BufferedWriter(new FileWriter(f));
			
			for (int s = 0; s < unlabeled_corpus.size(); s++) {	//all sentences
				Sentence sentence = unlabeled_corpus.get(s);
				Matrix logMatrix = new Matrix(num_postags, sentence.length());
				Matrix backMatrix = new Matrix(num_postags, sentence.length());
				Word begin = sentence.getWordAt(0);
				lemme = begin.getLemme();
				int lemmeNum = vocabulary.get(lemme);
				
				for (int i = 0; i < num_postags; i++) {	//pi
					double prob = Math.log(pi.get(0, i)) + Math.log(B.get(i, lemmeNum));
					logMatrix.set(i, 0, prob);
					backMatrix.set(i, 0, 0);
				}
				
				for (int j = 1; j < sentence.length(); j++) {
					Word w = sentence.getWordAt(j);
					lemme = w.getLemme();
					for (int i = 0; i < num_postags; i++) {	//rows of column j
						double max = -99999;
						int back = 0;
						for (int k = 0; k < num_postags; k++) {	//rows of column j-1
							double prob = logMatrix.get(k, j-1) + 
									Math.log(A.get(k, i)) + 
									Math.log(B.get(i, vocabulary.get(lemme)));
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
				for (int m = sentence.length()-1; m >0; m--) {
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
		
		for(int time = 0; time < 30; time++) {
			
			Matrix aUp = new Matrix(num_postags, num_postags, 0);	//numerator of aij
			Matrix aDown = new Matrix(1, num_postags, 0);	//denominator of aij
			Matrix bUp = new Matrix(num_postags, num_words, 0);	//numerator of bk(j)
			Matrix bDown = new Matrix(1, num_postags, 0);	//denominator of bk(j)
			Matrix sentencePi = new Matrix(1, num_postags, 0);
			
			for (int s = 0; s < unlabeled_corpus.size(); s++) {	//all sentences
				
				Sentence sentence = unlabeled_corpus.get(s);
				double[][][] xi = new double[num_postags][num_postags][sentence.length()-1];
				Matrix gamma = new Matrix(num_postags, sentence.length(), 0);
				Matrix alpha = alphas.get(s);
				Matrix beta = betas.get(s);
				
				//e
				for (int i = 0; i < num_postags; i++) {
					for (int j = 0; j < num_postags; j++) {
						for(int t = 0; t < sentence.length()-1; t++) {
							Word w = sentence.getWordAt(t+1);
							String lemme = w.getLemme();
							double prob = alpha.get(i, t)*beta.get(j, t+1)*A.get(i, j)*B.get(j, vocabulary.get(lemme));
							xi[i][j][t] = prob;
						}
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
						double aUpSum = 0;
						for(int t = 0; t < sentence.length()-1; t++) {
							aUpSum += xi[i][j][t];
							aDownSum += xi[i][j][t];
						}
						aUp.set(i, j, aUp.get(i, j)+aUpSum);
					}
					aDown.set(0, i, aDown.get(0, i)+aDownSum);
				}
				
				
				for (int i = 0; i < num_postags; i++) {
					double bDownSum = 0;
					for(int t = 0; t < sentence.length(); t++) {
						Word w = sentence.getWordAt(t);
						String lemme = w.getLemme();
						int index = vocabulary.get(lemme);
						double a = bUp.get(i, index);
						double b = gamma.get(i, t);
						bUp.set(i, index, a+b);
						
						bDownSum += gamma.get(i, t);
					}
					bDown.set(0, i, bDown.get(0, i) + bDownSum);
				}
				
				for(int i = 0; i < num_postags; i++) {
					sentencePi.set(0, i, sentencePi.get(0, i)+gamma.get(i, 0));
				}
					
			}	//end all sentences
			
			A = new Matrix(num_postags, num_postags, 0);
			B = new Matrix(num_postags, num_words, 0);
			pi = new Matrix(1, num_postags, 0);
			
			for(int i = 0; i < num_postags; i++) {
				for(int j = 0; j < num_postags; j++) {
					double prob = aUp.get(i, j)/aDown.get(0, i);
					A.set(i, j, prob+1.0E-90);
				}
				for(int t = 0; t < num_words; t++) {
					double prob = bUp.get(i, t)/bDown.get(0, i);
					B.set(i, t, prob+1.0E-90);
				}
				
				pi.set(0, i, sentencePi.get(0, i)/unlabeled_corpus.size()+1.0E-90);
			}
			
			forward();
			backward();
			
			if (time==29) {
				viterbi();
			}
			
		}	//end all 30 times
		
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
