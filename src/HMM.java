import java.util.Set;
import java.util.Hashtable;
import java.util.ArrayList;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;


import Jama.Matrix;

class HMM {
	/* Section for variables regarding the data */
	
	//
	private ArrayList<Sentence> labeled_corpus;
	
	//
	private ArrayList<Sentence> unlabeled_corpus;

	// number of pos tags
	int num_postags;
	
	// mapping POS tags in String to their indices
	Hashtable<String, Integer> pos_tags;
	
	// inverse of pos_tags: mapping POS tag indices to their String format
	Hashtable<Integer, String> inv_pos_tags;
	
	// vocabulary size
	int num_words;

	Hashtable<String, Integer> vocabulary;

	//similiar to tags, use 2 hashmap to store int - word mapping relationship. In order to retrieve the oi - word relation
	Map<String, Integer> word_tags;
	Map<Integer, String> inv_word_tags;


	Map<String, Integer> tagPairMap; // store the tag1+tag2 pair occurance
	Map<String, Integer> tagCount; // count tag occurance
	Map<String, Integer> tagWordPairMap; // store the tag + word pair occurance

	private int max_sentence_length;
	
	/* Section for variables in HMM */
	
	// transition matrix
	private Matrix A;

	// emission matrix
	private Matrix B;

	// prior of pos tags
	private Matrix pi;

	// store the scaled alpha and beta
	private Matrix alpha;
	
	private Matrix beta;

	// scales to prevent alpha and beta from underflowing
	private Matrix scales;

	// logged v for Viterbi
	private Matrix v;
	private Matrix back_pointer;
	private Matrix pred_seq;
	
	// \xi_t(i): expected frequency of pos tag i at position t. Use as an accumulator.
	private Matrix gamma;
	
	// \xi_t(i, j): expected frequency of transiting from pos tag i to j at position t.  Use as an accumulator.
	private Matrix digamma;
	
	// \xi_t(i,w): expected frequency of pos tag i emits word w.
	private Matrix gamma_w;

	// \xi_0(i): expected frequency of pos tag i at position 0.
	private Matrix gamma_0;
	
	/* Section of parameters for running the algorithms */

	// smoothing epsilon for the B matrix (since there are likely to be unseen words in the training corpus)
	// preventing B(j, o) from being 0
	private double smoothing_eps = 0.1;

	// number of iterations of EM
	private int max_iters = 10;

	// \mu: a value in [0,1] to balance estimations from MLE and EM
	// \mu=1: totally supervised and \mu = 0: use MLE to start but then use EM totally.
	private double mu = 0.8;
	
	/* Section of variables monitoring training */
	
	// record the changes in log likelihood during EM
	private double[] log_likelihood = new double[max_iters];
	
	/**
	 * Constructor with input corpora.
	 * Set up the basic statistics of the corpora.
	 */
	public HMM(ArrayList<Sentence> _labeled_corpus, ArrayList<Sentence> _unlabeled_corpus) {
		this.labeled_corpus = _labeled_corpus;
		this.unlabeled_corpus = _unlabeled_corpus;
		pos_tags = new Hashtable<>();
		inv_pos_tags = new Hashtable<>();
		vocabulary = new Hashtable<>();

		tagPairMap = new HashMap<>();
		tagWordPairMap = new HashMap<>();
		tagCount = new HashMap<>();



	}

	/**
	 * Set the semi-supervised parameter \mu
	 */
	public void setMu(double _mu) {
		if (_mu < 0) {
			this.mu = 0.0;
		} else if (_mu > 1) {
			this.mu = 1.0;
		}
		this.mu = _mu;
	}

	/**
	 * Create HMM variables.
	 */
	public void prepareMatrices() {
		int tagIdx = 0;
		int wordIdx = 0;
		int tempTagIdx = 0;
		for (Sentence tempSentence : labeled_corpus){
			for (int i = 0; i < tempSentence.length(); i++){
				//System.out.println(tempSentence.getWordAt(i).getLemme());
				Word tempWordClass = tempSentence.getWordAt(i);
				String tempWord = tempWordClass.getLemme();
				String tempTag = tempWordClass.getPosTag();
				//process new tag
				if (!pos_tags.containsKey(tempTag)){
					pos_tags.put(tempTag, tagIdx);
					inv_pos_tags.put(tagIdx, tempTag);
					tagIdx++;
				}
				//count tag
				if (!tagCount.containsKey(tempTag))
					tagCount.put(tempTag, 1);
				else
					tagCount.put(tempTag, tagCount.get(tempTag)+1);

				//process new word
				if (!word_tags.containsKey(tempWord)){
					word_tags.put(tempWord, wordIdx);
					inv_word_tags.put(wordIdx, tempWord);
					wordIdx++;
				}

				//count word
				if (!vocabulary.contains(tempWord)){
					vocabulary.put(tempWord, 1);
				}
				else{
					vocabulary.put(tempWord, vocabulary.get(tempWord) + 1);
				}

				//save pair tag
				if (i > 0){
					String previousWord = tempSentence.getWordAt(i-1).getLemme();
					String previousTag = tempSentence.getWordAt(i-1).getPosTag();
					String tagPair = previousTag + " " + tempTag;
					if (!tagPairMap.containsKey(tagPair))
						tagPairMap.put(tagPair, 1);
					else
						tagPairMap.put(tagPair, tagPairMap.get(tagPair)+1);

					//wordpair temporarily not usefull
				}

				//save word pair tag
				String tagWordPair = tempWord + " " + tempTag;
				if (!tagWordPairMap.containsKey(tagWordPair))
					tagWordPairMap.put(tagWordPair, 1);
				else
					tagWordPairMap.put(tagWordPair, tagWordPairMap.get(tagWordPair)+1);


				//System.out.println(tempWord + " " + tempTag + " " + tagIdx);

			}
		}

		num_words = vocabulary.size();
		num_postags = pos_tags.size();


		//check result
		System.out.println("total words are " + num_words);
		System.out.println("total tags are " + num_postags);
		for (String s : tagCount.keySet()){
			System.out.println(s + " " + tagCount.get(s));
		}
		for (String s : tagPairMap.keySet()){
			System.out.println(s + " " + tagPairMap.get(s));
		}

		//calculate mle
		mle();


	}

	/** 
	 *  MLE A, B and pi on a labeled corpus
	 *  used as initialization of the parameters.
	 */
	public void mle() {
		//cal A
		double[][] arrayA = new double[num_postags][num_postags];
		for (int i = 0; i < arrayA.length; i++){
			for (int j = 0; j < arrayA[0].length; j++){
				//aij = pair(qi&qj) / qi
				String qiqj = inv_pos_tags.get(i) + " " + inv_pos_tags.get(j);
				int countqiqj = tagPairMap.getOrDefault(qiqj, 0);
				int countqi = tagCount.get(inv_pos_tags.get(i));
				arrayA[i][j] = countqiqj / (double) countqi;
			}
		}

		A = new Matrix(arrayA);

		//cal B
		double[][] arrayB = new double[num_postags][num_words];
		for (int i = 0; i < arrayB.length; i++){
			for (int j = 0; j < arrayB[0].length; j++){
				String qioj = inv_pos_tags.get(i) + " " + inv_word_tags.get(j);
				int countqioj = tagWordPairMap.getOrDefault(qioj, 0);
				int countqi = tagCount.get(inv_pos_tags.get(i));
				arrayB[i][j] = countqioj / (double) countqi;
			}
		}
		B = new Matrix(arrayB);


		//cal pi. # of sentences is size of labeled_corpus
		int num_sentences = labeled_corpus.size();
		double[] arrayPi = new double[num_postags];
		for (int i = 0 ; i < arrayPi.length; i++){
			arrayPi[i] = tagCount.get(inv_pos_tags.get(i)) / (double) num_sentences;
		}

		pi = new Matrix(arrayPi);


	}

	/**
	 * Main EM algorithm. 
	 */
	public void em() {
	}
	
	/**
	 * Prediction
	 * Find the most likely pos tag for each word of the sentences in the unlabeled corpus.
	 */
	public void predict() {
	}
	
	/**
	 * Output prediction
	 */
	public void outputPredictions(String outFileName) throws IOException {
	}
	
	/**
	 * outputTrainingLog
	 */
	public void outputTrainingLog(String outFileName) throws IOException {
	}
	
	/**
	 * Expectation step of the EM (Baum-Welch) algorithm for one sentence.
	 * \xi_t(i,j) and \xi_t(i) are computed for a sentence
	 */
	private double expectation(Sentence s) {
		return 0;
	}

	/**
	 * Maximization step of the EM (Baum-Welch) algorithm.
	 * Just reestimate A, B and pi using gamma and digamma
	 */
	private void maximization() {
	}

	/**
	 * Forward algorithm for one sentence
	 * s: the sentence
	 * alpha: forward probability matrix of shape (num_postags, max_sentence_length)

	 * return: log P(O|\lambda)
	 */
	private double forward(Sentence s) {
		return 0;
	}

	/**
	 * Backward algorithm for one sentence
	 * 
	 * return: log P(O|\lambda)
	 */
	private double backward(Sentence s) {
		return 0;
	}

	/**
	 * Viterbi algorithm for one sentence
	 * v are in log scale, A, B and pi are in the usual scale.
	 */
	private double viterbi(Sentence s) {
		return 0;
	}

	public static void main(String[] args) throws IOException {
		if (args.length < 3) {
			System.out.println("Expecting at least 3 parameters");
			System.exit(0);
		}
		String labeledFileName = args[0];
		String unlabeledFileName = args[1];
		String predictionFileName = args[2];
		
		String trainingLogFileName = null;

		
		if (args.length > 3) {
			trainingLogFileName = args[3];
		}
		
		double mu = 0.0;
		
		if (args.length > 4) {
			mu = Double.parseDouble(args[4]);
		}
		// read in labeled corpus
		FileHandler fh = new FileHandler();
		
		ArrayList<Sentence> labeled_corpus = fh.readTaggedSentences(labeledFileName);
		
		ArrayList<Sentence> unlabeled_corpus = fh.readTaggedSentences(unlabeledFileName);

		HMM model = new HMM(labeled_corpus, unlabeled_corpus);
		
		model.setMu(mu);
		
		model.prepareMatrices();
		
		model.em();
		model.predict();
		model.outputPredictions(predictionFileName + "_" + String.format("%.1f", mu) + ".txt");
		
		if (trainingLogFileName != null) {
			model.outputTrainingLog(trainingLogFileName + "_" + String.format("%.1f", mu) + ".txt");
		}
	}
}
