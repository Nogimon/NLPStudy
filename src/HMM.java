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
	private double[][] alpha1;
	
	private Matrix beta;
	private double[][] beta1;

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

		word_tags = new HashMap<>();
		inv_word_tags = new HashMap<>();
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
		//pos_tags.put("START", 0);
		//inv_pos_tags.put(0, "START");
		//tagCount.put("START", 0);
		for (Sentence tempSentence : labeled_corpus){
			//process start
			String previousTag= "START";
			//tagCount.put("START", tagCount.get("START") + 1);
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
					previousTag = tempSentence.getWordAt(i-1).getPosTag();
				}
				String tagPair = previousTag + " " + tempTag;
				if (!tagPairMap.containsKey(tagPair))
					tagPairMap.put(tagPair, 1);
				else
					tagPairMap.put(tagPair, tagPairMap.get(tagPair)+1);

					//wordpair temporarily not usefull
				

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
		/*
		for (String s : tagCount.keySet()){
			System.out.println(s + " " + tagCount.get(s));
		}
		for (String s : tagPairMap.keySet()){
			System.out.println(s + " " + tagPairMap.get(s));
		}
		*/

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
		double[][] arrayB = new double[num_postags][num_words+1];
		for (int i = 0; i < arrayB.length; i++){
			for (int j = 0; j < arrayB[0].length-1; j++){
				String qioj =  inv_word_tags.get(j) + " " + inv_pos_tags.get(i);
				int countqioj = tagWordPairMap.getOrDefault(qioj, 0);
				int countqi = tagCount.get(inv_pos_tags.get(i));
				arrayB[i][j] = countqioj / (double) countqi;
			}
			arrayB[i][arrayB[0].length-1] = 0.1;
		}
		B = new Matrix(arrayB);


		//cal pi. # of sentences is size of labeled_corpus
		int num_sentences = labeled_corpus.size();
		double[] arrayPi = new double[num_postags];
		for (int i = 0 ; i < arrayPi.length; i++){
			arrayPi[i] = tagCount.get(inv_pos_tags.get(i)) / (double) num_sentences;
		}

		pi = new Matrix(arrayPi, 1);
		//pi size : (1, 44)

	}

	/**
	 * Main EM algorithm. 
	 */
	public void em(){
		for (int i = 1; i <= 3; i++){
			calEM();
			System.out.println("finished EM for the " + i + " round");
		}
	}


	public void calEM() {
		//A,B is already preset
		double[][] Aarray = A.getArray();
		double[][] Barray = B.getArray();
		//for (int i = 0; i < Aarray.length; i++){
		//	System.out.println(Arrays.toString(Aarray[i]));
		//}

		double[][] ahat = new double[num_postags][num_postags];// = sum(ksi) over t, not sum over j yet
		double[][] bhat = new double[num_postags][num_words+1]; // = sum(gamma) for j and o.


		for (Sentence s : unlabeled_corpus){
			//get alpha and beta, size : s.length()
			double probf = forward(s);
			double probb = backward(s);


			//double sumC = 0;
			//for (int j = 0; j < alpha1[s.length()-1].length; j++){
			//	sumC += alpha1[s.length()-1][j];
			//}
			double P = 1;

			//xi(i, j) = alpha(i) * aij * bjo * beta(j) / P(O|lambda)
			double[][] ksi = new double[num_postags][num_postags];
			double[] gamma = new double[num_postags];


			for (int i = 0; i < num_postags; i++){
				for (int j = 0; j < num_postags; j++){
					double sumxi = 0;
					for (int t = 0; t < s.length()-1; t++){
						//first get o
						int o;
						if (!word_tags.containsKey(s.getWordAt(t+1).getLemme())){
							o = num_words;
						}
						else {
							o = word_tags.get(s.getWordAt(t+1).getLemme());
						}
						//calculate (6.38)
						double temp = alpha1[t][i] * Aarray[i][j] * Barray[j][o] * beta1[t+1][j];
						sumxi /= P;
						sumxi += temp;

						double tempgamma = alpha1[t][j] * beta1[t][j];
						tempgamma /= P;
						bhat[j][o] += tempgamma;

					}

					//now sumxi is the numerator
					ksi[i][j] = sumxi;
					ahat[i][j] += ksi[i][j];




				}
			}



		}


		//now update A and B
		for (int i = 0; i < num_postags; i++){
			//first calulate denominator for a hat
			double denom = 0;
			for (int j = 0; j < num_postags; j++){
				denom += ahat[i][j];
			}
			//then update a
			for (int j = 0; j < num_postags; j++) {
				Aarray[i][j] = ahat[i][j] / denom;
			}
		}

		for (int j = 0; j < num_postags; j++){
			//first calculate denom
			double denom = 0;
			for (int o = 0; o < num_words+1; o++){
				denom += bhat[j][o];
			}
			//then update b
			for (int o = 0; o < num_words+1; o++){
				Barray[j][o] = bhat[j][0] / denom;
			}
		}


		A = new Matrix(Aarray);
		B = new Matrix(Barray);

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


	//implement forward algorithm for project Phase1
	private void forwardAlgorithm(String outputDirect) throws IOException{
		FileWriter fw = new FileWriter(outputDirect);
		BufferedWriter bw = new BufferedWriter(fw);
		for (Sentence s : unlabeled_corpus){
			double prob = forward(s);
			bw.write(String.valueOf(prob));
			bw.write("\n");
				
		}
		bw.close();
		fw.close();
	}

	/**
	 * Forward algorithm for one sentence
	 * s: the sentence
	 * alpha: forward probability matrix of shape (num_postags, max_sentence_length)

	 * return: log P(O|\lambda)
	 */
	private double forward(Sentence s) {
		double[] arrayAlpha = new double[num_postags];
		Matrix alpha = new Matrix(arrayAlpha, 1);//1 row, arrayAlpha.length colums
		alpha1 = new double[s.length()][num_postags];//for member variable alpha[t][j] -> a_t(j)
		double P = 0;
		for (int i = 0; i < s.length(); i++){
			double sum = 0;
			int o;
			if (!word_tags.containsKey(s.getWordAt(i).getLemme())){
				o = num_words;
			}
			else {
				o = word_tags.get(s.getWordAt(i).getLemme());
			}
			if (i == 0){//first word : pi(j) * bjo
				for (int j = 0; j < arrayAlpha.length; j++){
					arrayAlpha[j] = pi.get(0, j) * B.get(j, o);
					sum += arrayAlpha[j];
				}
			}
			else{//other word: alpha * a * b
				for (int j = 0; j < arrayAlpha.length; j++){
					Matrix temp2 = A.getMatrix(j, j, 0, num_postags-1);
					Matrix temp3 = A.getMatrix(0, num_postags-1, j, j);
					Matrix temp = alpha.times(temp3);
					arrayAlpha[j] = temp.get(0,0)  * B.get(j, o);
					//System.out.print(arrayAlpha[j] + " ");
					sum += arrayAlpha[j];
				}
			}

			//System.out.println("sum is " + sum + " log 1/sum is " + Math.log(1/sum));
			alpha = new Matrix(arrayAlpha, 1);
			alpha = alpha.times(1/sum);

			for (int j = 0; j < num_postags; j++){
				alpha1[i][j] = alpha.get(0,j);
			}
			//System.out.println(Arrays.toString(alpha1[i]));

			//logP = - sum(log c)
			P -= Math.log(1/sum);
		}

		return P;
	}

	/**
	 * Backward algorithm for one sentence
	 * 
	 * return: log P(O|\lambda)
	 */

	private double backward(Sentence s) {
		double[] arrayBeta = new double[num_postags];
		Matrix beta = new Matrix(arrayBeta, 1);
		beta1 = new double[s.length()][num_postags];
		double P = 0;
		for (int i = s.length()-1; i >= 0; i--){
			double sum = 0;
			int o;
			if (!word_tags.containsKey(s.getWordAt(i).getLemme())){
				o = num_words;
			}
			else {
				o = word_tags.get(s.getWordAt(i).getLemme());
			}
			if (i == s.length()-1){//last word : ct-1 = 1/sum(alphat)
				double sumC = 0;
				for (int j = 0; j < alpha1[s.length()-1].length; j++){
					sumC += alpha1[s.length()-1][j];
				}
				sumC = 1/sumC;
				for (int j = 0; j < arrayBeta.length; j++){
					arrayBeta[j] = sumC;//pi.get(0, j) * B.get(j, o);
					sum += arrayBeta[j];
				}
			}
			else{//other word: beta * a * b
				for (int j = 0; j < arrayBeta.length; j++){
					Matrix temp3 = A.getMatrix(0, num_postags-1, j, j);
					Matrix temp = beta.times(temp3);
					arrayBeta[j] = temp.get(0,0)  * B.get(j, o);
					//System.out.print(arrayBeta[j] + " ");
					sum += arrayBeta[j];
				}
			}

			//System.out.println("sum is " + sum + " log 1/sum is " + Math.log(1/sum));
			beta = new Matrix(arrayBeta, 1);
			beta = beta.times(1/sum);

			for (int j = 0; j < num_postags; j++){
				beta1[i][j] = beta.get(0,j);
			}

			//logP = - sum(log c)
			P -= Math.log(1/sum);
		}



		return P;
	}




	//viterbi algorithm for phase 1
	private void viterbiAlgorithm(String outputDirect) throws IOException{
		FileWriter fw = new FileWriter(outputDirect);
		BufferedWriter bw = new BufferedWriter(fw);
		for (Sentence s : unlabeled_corpus){
			int[] sequence = viterbi(s);
			for (int i = 0; i < sequence.length; i++){
				bw.write(s.getWordAt(i).getLemme());
				bw.write(" ");
				bw.write(inv_pos_tags.get(sequence[i]));
				bw.write("\n");
			}
			bw.write("\n");

		}
		bw.close();
		fw.close();
	}
	/**
	 * Viterbi algorithm for one sentence
	 * v are in log scale, A, B and pi are in the usual scale.
	 */
	private int[] viterbi(Sentence s) {
		double[] arrayV = new double[num_postags];
		Matrix matrixV = new Matrix(arrayV, 1);
		int[] tagSequence = new int[s.length()];
		double currmax = 0;
		for (int i = 0; i < s.length(); i++){
			int o;
			if (!word_tags.containsKey(s.getWordAt(i).getLemme())){
				o = num_words;
			}
			else {
				o = word_tags.get(s.getWordAt(i).getLemme());
			}
			//currmax = 0;
			int maxlabel = 0;
			//first word : pi(j) * bjo and get max
			if (i == 0){
				for (int j = 0; j < arrayV.length; j++){
					arrayV[j] = pi.get(0, j)  * B.get(j, o);
					if (arrayV[j] > currmax) {
						currmax = arrayV[j];
						maxlabel = j;
					}
				}

			}
			//other word: vi * aij * bjo, and get max
			else{
				for (int j = 0; j < arrayV.length; j++){
					Matrix temp3 = A.getMatrix(0, num_postags-1, j, j).transpose();
					Matrix temp = matrixV.arrayTimes(temp3);
					double[] tempArray = temp.getArray()[0];
					double maxForOneQ = 0; //this is the max for one state qj
					for (int k = 0; k < tempArray.length; k++){
						maxForOneQ = Math.max(maxForOneQ, tempArray[k]);
					}


					arrayV[j] = maxForOneQ  *B.get(j, o);
					if (arrayV[j] > currmax){
						currmax = arrayV[j];
						maxlabel = j;
					}
				}

			}

			tagSequence[i] = maxlabel;
			//System.out.println("the tag is " + maxlabel + " its v is " + currmax);
			currmax = 0;
			matrixV = new Matrix(arrayV, 1);

		}

		return tagSequence;
	}

	private int[] viterbilog(Sentence s) {
		double[] arrayV = new double[num_postags];
		double[] narrayV = new double[num_postags];
		int[] tagSequence = new int[s.length()];
		double currmax = 0;
		for (int i = 0; i < s.length(); i++){
			int o;
			if (!word_tags.containsKey(s.getWordAt(i).getLemme())){
				o = num_words;
			}
			else {
				o = word_tags.get(s.getWordAt(i).getLemme());
			}
			//currmax = 0;
			int maxlabel = 0;
			//first word : pi(j) * bjo and get max
			if (i == 0){
				for (int j = 0; j < arrayV.length; j++){
					arrayV[j] = Math.log(pi.get(0, j) ) + Math.log( B.get(j, o));
					if (arrayV[j] > currmax) {
						currmax = arrayV[j];
						maxlabel = j;
					}
				}

			}
			//other word: vi * aij * bjo, and get max
			else{
				narrayV = new double[num_postags];
				for (int j = 0; j < arrayV.length; j++){
					//get transition matrix column j
					Matrix temp3 = A.getMatrix(0, num_postags-1, j, j).transpose();
					double[] aij = temp3.getArray()[0];
					//max each transition from previous position k
					for (int k = 0; k < aij.length; k++){
						double curr = arrayV[k] + Math.log(aij[k]);
						narrayV[j] = Math.max(narrayV[j], curr);
					}
					//plus bjo
					narrayV[j] += Math.log(B.get(j, o));
					if (arrayV[j] > currmax){
						currmax = arrayV[j];
						maxlabel = j;
					}
				}

			}

			tagSequence[i] = maxlabel;
			System.out.println("the tag is " + maxlabel + " its v is " + currmax);
			currmax = 0;
			arrayV = narrayV;

		}

		return tagSequence;
	}


	public static void main(String[] args) throws IOException {
		/*
		if (args.length < 3) {
			System.out.println("Expecting at least 3 parameters");
			System.exit(0);
		}
		*/
		/*
		String labeledFileName = args[0];
		String unlabeledFileName = args[1];
		String predictionFileName = args[2];
		*/
		String trainingLogFileName = null;

		String labeledFileName = "./data/p2/train.txt";
		String unlabeledFileName = "./data/p2/test.txt";
		String predictionFileName = "./results/p2/result.txt";



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

		model.forwardAlgorithm("./results/p2/log.txt");
		model.viterbiAlgorithm("./results/p2/predictions.txt");

		model.em();
		model.predict();
		model.outputPredictions(predictionFileName + "_" + String.format("%.1f", mu) + ".txt");
		
		if (trainingLogFileName != null) {
			model.outputTrainingLog(trainingLogFileName + "_" + String.format("%.1f", mu) + ".txt");
		}
	}
}
