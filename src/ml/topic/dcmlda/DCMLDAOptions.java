package ml.topic.dcmlda;

// If not set by users, just use these default values. 
public class DCMLDAOptions {
    public double alpha = 1;     // alpha is the prior for document-topic distributions
    public double beta = 0.01;   // beta is the prior for topic-word distributions. 
    public int burnin = 100;     // number of iterations before we start sampling
    public int sampleWait = 10;  // number of iterations for per sampling period
    public int samples = 10;     // number of sampling period
    public int tWords = 20;      // number of top words to display
    public int K = 50;           // number topics we learn from the model.
    public String inputFile = "data/nipstxt"; // by default, use this for testing. 
}
