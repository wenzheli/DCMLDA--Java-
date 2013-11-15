package ml.topic.dcmlda;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

import ml.topic.common.data.DataSet;
import ml.topic.common.preprocess.DataSetGen;
import ml.topic.utils.QuickSort;

/**
 * @author wenzhe
 */
public class DCMLDAInference {
    DataSet dataset;
    DCMLDAModel model;
    DCMLDAOptions option;
    
    public DCMLDAInference(DataSet dataset){
        this.dataset = dataset;
    }
    
    public void initModel(DCMLDAOptions options){
        model = new DCMLDAModel();
        model.init(options, dataset);
        this.option = options;
    }
    

    public void runSampler() throws FileNotFoundException, UnsupportedEncodingException{
        // burn-in period
        System.out.println("start burn-in period.......");
        for (int itr = 0 ; itr < option.burnin ; itr++){
            System.out.println("gibbs sampling: " + itr + " iteration");
            model.runGibbsSampler();
        }
        System.out.println("finished burn-in period, move to sampling period.......");
        
        // sampling period
        for (int itr = 0 ; itr < option.samples ; itr++){
            System.out.println("sampling period: " + itr);
            // first run gibbs sampler..
            for (int i = 0; i < option.sampleWait; i++){
                System.out.println("gibbs sampling: " + i + " iteration");
                model.runGibbsSampler();  
            }
            // updating alpha and beta..
            System.out.println("updating hyperparameters");
            model.updateHyperparameters(itr);
            printTopWords(itr);
        }
     
    }
    
    /**
     * Print the top words on the console, also write it into file. 
     */
    public void printTopWords(int itr) throws FileNotFoundException, UnsupportedEncodingException{
        PrintWriter writer = new PrintWriter("result/DCMLDA/" + "_topWords_"+itr+".txt", "UTF-8");
        System.out.println("Printing top words.....");
        writer.println("Printing top words....");
        
        // beta is the prior for DCMLDA, which plays similar role as \psi in LDA. 
        double[][] beta = model.getBeta();
        
        int len1 = beta.length;
        int len2 = beta[0].length;
        double[][] temp = new double[len1][len2];
        for (int i = 0; i < len1; i++){
            for (int j =0; j < len2; j++){
                temp[i][j] = beta[i][j];
            }
        }
        
        int tTop = option.tWords; // get the tTop words from each topic
        String[][] topWords = new String[option.K][tTop];
        double[][] topWordsProbability = new double[option.K][tTop];
        
        for (int k = 0; k < option.K; k++){
            // select the top words for topic k
            int vocabSize = dataset.vocab.getVocabularySize();
            System.out.println("Top words for topic: " + k);
            
            
            int[] index = new int[vocabSize];
            for (int v = 0; v < vocabSize; v++){
                index[v] = v;
            }
            
            QuickSort.quicksort(temp[k], index);
           
            for (int i = 0; i < tTop; i++){
                topWords[k][i] = dataset.vocab.indexTotokenMap.get(index[vocabSize-i-1]); 
                topWordsProbability[k][i] = beta[k][index[vocabSize-i-1]];
            }
        }
        
        for (int k = 0; k < option.K; k++){
            writer.println("Top words for topic: " + k);
            System.out.println("Top words for topic: " + k);
            for (int i = 0; i < topWords[k].length; i++){
                System.out.println(topWords[k][i] + ":     " + topWordsProbability[k][i]);
                writer.println(topWords[k][i] + ":     " + topWordsProbability[k][i]);
            }
            
            System.out.println("****************************************");
            writer.println("****************************************");
        }
        
        writer.close();
    }
    
    /**
     * Run DCMLDA for document corpus. Try run this example, you don't need to pass any argument
     * for making this work. 
     * 
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException{
        System.out.println("-nTopics: number of topics");
        System.out.println("-alpha: default value for alpha prior");
        System.out.println("-beta: default value for beta prior");
        System.out.println("-burnin: number of iterations before sampling starts");
        System.out.println("-samples: number of samples for estimating parameters");
        System.out.println("-sampleWait: number of iterations for each sampling period");
        System.out.println("-tWords: number of top words for each topic");
        System.out.println("-file: input data file");
        
        if (args.length % 2 != 0){
            System.err.println("The input arguments are not in correct format!");
        }
        DCMLDAOptions opt = new DCMLDAOptions();
        for (int i = 0; i < args.length; i++){
            if (args[i].equals("-nTopics")){
                opt.K = Integer.parseInt(args[i+1]);
            } else if (args[i].equals("-alpha")){
                opt.alpha = Double.parseDouble(args[i+1]);
            } else if (args[i].equals("-beta")){
                opt.beta = Double.parseDouble(args[i+1]);
            } else if (args[i].equals("-burnin")){
                opt.burnin = Integer.parseInt(args[i+1]);
            } else if (args[i].equals("-samples")){
                opt.samples = Integer.parseInt(args[i+1]);
            } else if (args[i].equals("-sampleWait")){
                opt.sampleWait = Integer.parseInt(args[i+1]);
            } else if (args[i].equals("-tWords")){
                opt.tWords = Integer.parseInt(args[i+1]);
            } else if (args[i].equals("-file")){
                opt.inputFile = args[i+1];
            }
            i++;
        }
        
        DataSet dataset = DataSetGen.createFromNIPSCorpus("data/nipstxt");
        DCMLDAInference inference = new DCMLDAInference(dataset);
       
        inference.initModel(opt);
        inference.runSampler();
    }
}
