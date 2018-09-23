import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.*;
import java.util.*;

import java.util.List;


public class Predictor {

    /**
      * args[0]: test tokens
      * args[1]: output direction
      */
    public static void main(String[] args) throws IOException {
        String inputFileName = args[0];
        String outputFileName = args[1];

        //save all the tokens to be checked
        String[][] confuse = {{"accept", "except"},{"adverse", "averse"},{"advice","advise"},{"affect","effect"},{"aisle","isle"},{"aloud", "allowed"},{"altar","alter"},{"amoral","immoral"},{"appraise","apprise"},{"assent","ascent"}};
        List<String> confuseSet = new ArrayList<>();
        for (String[] ss : confuse){
            confuseSet.add(ss[0]);
            confuseSet.add(ss[1]);
        }

        //read all the tokesn pair probablilities

        BufferedReader br = new BufferedReader(new FileReader("results/pairprob.txt"));
        Map<String, Double> confusemap = new HashMap<>();

        String line = " ";
        while(line != null){
            line = br.readLine();
            line = line.split(" ");
            confusemap.put(line[0] + " " + line[1], Double.parseDouble(line[2]);

        }
        br.close();



        br = new BufferedReader(new FileReader(inputFileName));

        line = br.readLine();
        String previous;

        //store single word set
        Set<String> setv = new HashSet<>();
        //store pair word appearance
        Map<String, Integer> map = new HashMap<>();

        while(line!=null){
            line = br.readLine();

            
            if (!setv.contains(line)){
                setv.add(line);
            }
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

        //Write all the pairs into bigram.txt
        FileWriter fw = new FileWriter("data/bigram.txt");
        BufferedWriter bw = new BufferedWriter(fw);
        for (String s : map.keySet()){
            bw.write(s + " " + map.get(s) + "\n");
        }
        bw.close();
        fw.close();


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
        fw = new FileWriter("data/ff.txt");
        bw = new BufferedWriter(fw);
        for (int count : nMap.keySet()){
            bw.write(Math.log(count) + " " + Math.log(nMap.get(count)) + "\n");
            if (count <= 10)
                nc[count-1] = nMap.get(count);
        }
        bw.close();
        fw.close();


        //calculate GT smoothing for 1 to 5 and save in GTTable.txt
        fw = new FileWriter("data/GTTable.txt");
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

    }
}
