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

        /*
        //For pair check
        //This part is an addition to the predictor part, not finished yet by the ddl,
        //but could be usefull if the previous word information is used

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
        */

        //Read all the pair decisions
        BufferedReader br = new BufferedReader(new FileReader("results/decision.txt"));
        Map<String, String> decisionMap = new HashMap<>();
        String line = br.readLine();
        while(line != null){
            String[] lines = line.split(" ");
            decisionMap.put(lines[0], lines[1]);
            line = br.readLine();
        }
        br.close();




        //Read the file, and gnenrate the prediction file
        br = new BufferedReader(new FileReader(inputFileName));
        FileWriter fw = new FileWriter(outputFileName);
        BufferedWriter bw = new BufferedWriter(fw);

        line = br.readLine();;
        int linenum = 0;
        int tokennum = 0;
        boolean newline = true;

        while(line!=null){
            
            tokennum++;
            if (line.equals("<s>")){
                linenum++;
                tokennum = 0;
                newline = true;
            }
            if (line.equals("</s>") && newline == false)
                bw.write("\n");

            if (decisionMap.containsKey(line)){

                if (newline == true){
                    bw.write(linenum +":" + tokennum + ",");
                    newline = false;
                }
                else{
                    bw.write(tokennum + ",");
                }
            }
            line = br.readLine();
        }

        br.close();
        bw.close();
        fw.close();


        

    }
}
