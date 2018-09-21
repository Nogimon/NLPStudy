import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.*;
import java.util.*;

import java.util.List;


/**
  * This class takes a text file as input and tokenize sentences and tokens.
  * The output is a stream of tokens, with <s> and </s> indicating sentence boundaries.
  */
public class ProbEstimator {

	/**
	  * args[0]: source text file
	  * args[1]: output tokenized file
	  */
	public static void main(String[] args) throws IOException {
		String inputFileName = args[0];//"/home/zlab-1/lian/course/project_1_release/data/train_reviews.txt"; //args[0];
		//String outputFileName = args[1];//"/home/zlab-1/lian/course/project_1_release/data/train_reviews.txt"; //args[1];

		//FileWriter fw = new FileWriter(outputFileName);
		//BufferedWriter bw = new BufferedWriter(fw);

		//DocumentPreprocessor dp = new DocumentPreprocessor(inputFileName);

		//Scanner in = new Scanner(new BufferedReader(new FileReader(inputFileName)));

		BufferedReader br = new BufferedReader(new FileReader(inputFileName));

		String line = br.readLine();

		Set<String> setv = new HashSet<>();
		Map<String, Integer> map = new HashMap<>();

		String previous = line;
		setv.add(previous);

		while(line!=null){
			line = br.readLine();
			if (!setv.contains(line)){
				setv.add(line);
			}
			String pair = previous + "#" + line;
			if (!map.containsKey(pair)){
				map.put(pair, 1);
			}
			else{
				map.put(pair, map.get(pair) + 1);
			}
		}

		br.close();

		int v = setv.size();
		System.out.println(v);
		System.out.println(map.size());

		//3.1 : N0 equals to all the possible wv s , minus the existing wv pairs : N0 = C(v, 2) - map.size()
		System.out.println("N0 is " + (v*(v-1)/2 - map.size()));

		//nMap to store <Nc, C>. first is the appearance time, second is how many that occurs at this time
		Map<Integer, Integer> nMap = new HashMap<>();

		int nBig = 0;//to calculate N: the total count of N
		for (int count : map.values()){
			nBig += count;
			if (!nMap.containsKey(count)){
				nMap.put(count, 1);
			}
			else{
				nMap.put(count, nMap.get(count) + 1);
			}
		}
		
		for (int count : nMap.keySet()){
			//System.out.println(count + " " + nMap.get(count));
			System.out.println(Math.log(count)+" " + Math.log(nMap.get(count)));
		}
		System.out.println("the total N is " + nBig);

	}
}
