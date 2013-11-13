package ml.topic.common.preprocess;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.topic.common.data.DataSet;
import ml.topic.common.data.Document;
import ml.topic.common.data.Vocabulary;
import ml.topic.utils.TextUtil;
import ml.topic.common.preprocess.PorterStemmer;
import ml.topic.common.preprocess.StopWords;

public class DataSetGen {  
    public final static int MIN_COUNT = 30;
    public final static int MAX_COUNT = 3000;
    public final static int MAX_FILE_COUNT = 2000;
    
    @SuppressWarnings("resource")
    public static DataSet createFromNIPSCorpus(String filePath) throws IOException{
        DataSet dataset = new DataSet();
        BufferedReader br = null;
        File directory = new File(filePath);
        File[] files = directory.listFiles();
        String sCurrentLine = "";
        
        // we will count the number for each token.
        Map<String, Integer> tokenMap = new HashMap<String, Integer>();
        // first iteration, creating vocabulary list. 
        int fileCount = 0;
        for (File folder: files){
            for (File f: folder.listFiles()){
                fileCount++;
                if (fileCount > MAX_FILE_COUNT)
                    break;     
                System.out.println("Processing " + ++fileCount + "th file");
                
                br = new BufferedReader(new FileReader(f));
                while ((sCurrentLine = br.readLine()) != null){
                        String text = sCurrentLine;
                        // remove the stop words
                        String[] tokens = text.split("\\s");
                        for (String token : tokens){
                            // 1. stemming the token. 
                            PorterStemmer stem = new PorterStemmer();
                            String stemmedToken = stem.stemming(token);
                            // 2. remove special characters. 
                            String processedToken = TextUtil.removeSpecialCharacters(stemmedToken);
                            if (processedToken.equals(""))
                                continue;
                            // 3. check if any invalid character is contained in the token
                            if (TextUtil.isValid(processedToken)){
                                continue;
                            }
                            // 4. finally, remove the stop words. 
                            if (StopWords.isStopword(processedToken))
                                continue;
                            
                            // build the tokenMap
                            if (tokenMap.containsKey(processedToken)){
                                int cnt = tokenMap.get(processedToken);
                                tokenMap.put(processedToken, cnt + 1);
                            } else{
                                tokenMap.put(processedToken, 1);
                            }
                        }  
                }
            }
        }
        
        System.out.println("Finished building token map for all the documents. ");
        System.out.println("Now, start building our vocabulary and dataset...");
        
        int index  = 0;
        Map<String, Integer> tokenToIndex = new HashMap<String, Integer>();
        Map<Integer, String> indexToToken = new HashMap<Integer, String>();
        for (String token: tokenMap.keySet()){
            int cnt = tokenMap.get(token);
            // if out of this range, we will ignore this token. 
            if (cnt >= MIN_COUNT && cnt <= MAX_COUNT){
                tokenToIndex.put(token, index);
                indexToToken.put(index, token);
                index++;
            }
        }
        
        // build vocabulary object. 
        Vocabulary vocab = new Vocabulary();
        vocab.setIndexTotokenMap(indexToToken);
        vocab.settokenToIndex(tokenToIndex);
        
        System.out.println("1. Finished creating vocabulary object.");
        System.out.println("******************************************"); 
        System.out.println("Generating documents objects using vocabulary");
        
        fileCount = 0;
        List<Document> documents = new ArrayList<Document>();
        for (File folder: files){
            for (File f: folder.listFiles()){
                fileCount++;
                if (fileCount > MAX_FILE_COUNT)
                    break;
                
                List<Integer> tokensInDoc = new ArrayList<Integer>();
                Document doc = new Document();
                System.out.println("Process " + ++fileCount + "th file");
               
                br = new BufferedReader(new FileReader(f));
                while ((sCurrentLine = br.readLine()) != null){
                        String text = sCurrentLine;
                        // remove the stop words
                        String[] tokens = text.split("\\s");
                        for (String token : tokens){
                            // 1. stemming the token. 
                            PorterStemmer stem = new PorterStemmer();
                            String stemmedToken = stem.stemming(token);
                            // 2. remove special characters. 
                            String processedToken = TextUtil.removeSpecialCharacters(stemmedToken);
                            if (processedToken.equals(""))
                                continue;
                            // 3. check if any invalid character is contained in the token
                            if (TextUtil.isValid(processedToken)){
                                continue;
                            }
                            // 4. finally, remove the stop words. 
                            if (StopWords.isStopword(processedToken))
                                continue;
                            
                            if (tokenToIndex.containsKey(token)){
                                int tokenIndex = tokenToIndex.get(token);
                                tokensInDoc.add(tokenIndex);
                            }
                        } 
                }
                
                doc.setTokens(tokensInDoc);    // set tokens for document
                documents.add(doc);            // add curr doc to document list. 
            } 
        }
        
        dataset.setDocuments(documents);
        dataset.setVocabulary(vocab);
        System.out.println("Successfully process all the documents");   
        
        return dataset;
    }
}
