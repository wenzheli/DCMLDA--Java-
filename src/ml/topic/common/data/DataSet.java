package ml.topic.common.data;

import java.util.List;

/**  
 * @author wenzhe
 */
public class DataSet{
    List<Document> documents;
    public Vocabulary vocab;
   
    public List<Document> getDocuments(){
        return documents;
    }
    
    public Document getDocument(int index){
        return documents.get(index);
    }
    
    public void setDocuments(List<Document> documents){
        this.documents = documents;
    }
    
    public Vocabulary getVocabulary(){
        return vocab;
    }
    
    public void setVocabulary(Vocabulary vocab){
        this.vocab = vocab;
    }
    
    public int getDocumentCount(){
        return documents.size();
    }   
}
