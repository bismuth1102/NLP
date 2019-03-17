/**
 * Generate sentences from a CFG
 * 
 * @author sihong
 *
 */

import java.io.*;
import java.util.*;

public class Generator {
	
	private Grammar grammar;

	/**
	 * Constructor: read the grammar.
	 */
	public Generator(String grammar_filename) {
		grammar = new Grammar(grammar_filename);
	}

	/**
	 * Generate a number of sentences.
	 */
	public ArrayList<String> generate(int numSentences) {
	}
	
	public static void main(String[] args) {
		// the first argument is the path to the grammar file.
		Generator g = new Generator(args[0]);
		ArrayList<String> res = g.generate(1);
		for (String s : res) {
			System.out.println(s);
		}
	}
}
