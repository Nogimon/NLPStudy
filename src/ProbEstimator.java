import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.*;
import java.util.*;

import java.util.List;


public class ProbEstimator {

	/**
	  * args[0]: source text file
	  * args[1]: output tokenized file
	  */
	public static void main(String[] args) throws IOException {
		String inputFileName = args[0];//"/home/zlab-1/lian/course/project_1_release/data/train_reviews.txt"; //args[0];
		//String outputFileName = args[1];//"/home/zlab-1/lian/course/project_1_release/data/train_reviews.txt"; //args[1];


		//Part1 : process train tokens
		BufferedReader br = new BufferedReader(new FileReader(inputFileName));

		String line = br.readLine();

		//store single word set
		Set<String> setv = new HashSet<>();
		//Store all single word count
		Map<String, Integer> vmap = new HashMap<>();

		setv.add(line);
		vmap.put(line, 1);

		//store pair word appearance
		Map<String, Integer> map = new HashMap<>();

		String previous = line;
		while(line!=null){
			line = br.readLine();
			if (line == null) break;

			//line = line.replaceAll("[\\W+]", "");
			if (!setv.contains(line)){
				setv.add(line);
			}
			if (!vmap.containsKey(line))
				vmap.put(line, 1);
			else
				vmap.put(line, vmap.get(line)+1);

			//I saved previous ahead of line. that is (w,v) -> saved as "v w"
			String pair = previous + " " + line;
			if (!map.containsKey(pair)){
				map.put(pair, 1);
			}
			else{
				map.put(pair, map.get(pair) + 1);
			}
			previous = line;
		}

		br.close();


		//Part2 : Process spelling checks
		//save all the tokens to be checked
        String[][] confuse = {{"accept", "except"},{"adverse", "averse"},{"advice","advise"},{"affect","effect"},{"aisle","isle"},{"aloud", "allowed"},{"altar","alter"},{"amoral","immoral"},{"appraise","apprise"},{"assent","ascent"}};
        List<String> confuseSet = new ArrayList<>();
        for (String[] ss : confuse){
            confuseSet.add(ss[0]);
            confuseSet.add(ss[1]);
        }

		//Write all the pairs into bigram.txt, and save the data containing confusing pairs
		//confusemap : save <pairs, probability>
		Map<String, Double> confusemap = new HashMap<>();
		FileWriter fw = new FileWriter("results/bigram.txt");
		BufferedWriter bw = new BufferedWriter(fw);
		FileWriter fw2 = new FileWriter("results/pairprob.txt");
		BufferedWriter bw2 = new BufferedWriter(fw2);
		for (String s : map.keySet()){
			bw.write(s + " " + map.get(s) + "\n");
			String[] pairs = s.split(" ");
			//Note that the confuse word should appear later
			if (confuseSet.contains(pairs[1])){
				int cv = vmap.get(pairs[0]);
				double ppairs = map.get(s) / (double)cv;
				confusemap.put(s, ppairs);
				bw2.write(s + " " + ppairs + "\n");

			}
		}
		bw.close();
		fw.close();
		bw2.close();
		fw2.close();

		//Part3: Process all the parameters required in homework 3.1 etc
		int v = setv.size();
		System.out.println("V of one word is " + v);
		System.out.println("V of two words is " + map.size());

		//3.1 : N0 equals to all the possible wv s , minus the existing wv pairs : N0 = P(v, 2) - map.size()
		System.out.println("N0 is " + (v*(v-1) - map.size()));

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
		
		/*
		for (int count : nMap.keySet()){
			//System.out.println(count + " " + nMap.get(count));
			System.out.println(Math.log(count)+" " + Math.log(nMap.get(count)));
		}
		*/
		System.out.println("the total N is " + nBig);

		//count NC from 1 to 10.  for each i = C : nc[C] = NC.
		int[] nc = new int[10];

		//Write all log(C) and log(Nc) into ff.txt
		fw = new FileWriter("results/ff.txt");
		bw = new BufferedWriter(fw);
		for (int count : nMap.keySet()){
			bw.write(Math.log(count) + " " + Math.log(nMap.get(count)) + "\n");
			if (count <= 10)
				nc[count-1] = nMap.get(count);
		}
		bw.close();
		fw.close();


		//calculate GT smoothing for 1 to 5 and save in GTTable.txt
		fw = new FileWriter("results/GTTable.txt");
		bw = new BufferedWriter(fw);
		int nclimit = 5;
		bw.write("smooth upper limie is " + nclimit + "\n");
		for (int i = 0; i < nclimit; i++){
			System.out.println("N" + (i+1)+ " is " + nc[i] );
			double cstar = (i+1)*nc[i+1]/(double)nc[i];
			bw.write( i+ " " + cstar+ "\n");

		}
		bw.close();
		fw.close();

		double pl = 1/(double) (nBig + map.size());
		double pgt = nc[0] / (double)nBig;
		System.out.println("for zero frequency token\nLaplacian is " + pl + "\nGT is " + pgt);


		//Next decide which token to use in confusing pair
		Map<String, String> judgeMap = new HashMap<>();
		for (int i = 0; i < confuseSet.size(); i+=2){
			String s1 = confuseSet.get(i);
			String s2 = confuseSet.get(i+1);
			if (!vmap.containsKey(s1) && !vmap.containsKey(s2))
				continue;
			else if (vmap.containsKey(s1) && !vmap.containsKey(s2))
				judgeMap.put(s2, s1);
			else if (!vmap.containsKey(s1) && vmap.containsKey(s2))
				judgeMap.put(s1,s2);
			else{

				if (vmap.get(s1) > vmap.get(s2))
					judgeMap.put(s2, s1);
				else
					judgeMap.put(s1, s2);
			}
			
		}

		//Save the pair dicision into  dicision.txt
		fw = new FileWriter("results/decision.txt");
		bw = new BufferedWriter(fw);
		for (String s : judgeMap.keySet()){
			//System.out.println(s + " -> " + judgeMap.get(s));
			bw.write(s + " " + judgeMap.get(s)+ "\n");
		}
		bw.close();
		fw.close();



		



	}
}
