package ml.topic.common.data;

import java.util.HashMap;
import java.util.Map;

/**
 * Vocabulary class to store all the tokens from the dataset. 
 * @author wenzhe  
 *
 */
public class Vocabulary {
    
    public Map<String, Integer> tokenToIndexMap = new HashMap<String, Integer>();
    public Map<Integer, String> indexTotokenMap = new HashMap<Integer, String>();
        
    public void settokenToIndex(Map<String, Integer> tokenToIndexMap){
        this.tokenToIndexMap = tokenToIndexMap;
    }
    
    public void setIndexTotokenMap(Map<Integer, String> indexTotokenMap){
        this.indexTotokenMap = indexTotokenMap;
    }
    
    public int getVocabularySize(){
        return tokenToIndexMap.size();
    }    
}
