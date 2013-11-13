package ml.topic.common.data;

import java.util.List;

public class Document {
    List<Integer> tokens;
    
    public List<Integer> getTokens(){
        return tokens;
    }
    
    public int getToken(int index){
        return tokens.get(index);
    }
    
    public void setTokens(List<Integer> tokens){
        this.tokens = tokens;
    }
    
    public int getNumOfTokens(){
        return tokens.size();
    } 
}
