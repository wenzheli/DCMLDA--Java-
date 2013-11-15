package ml.topic.dcmlda;

import ml.topic.common.data.DataSet;
import ml.topic.common.data.Document;
import ml.topic.utils.ConvergeUtil;
import ml.topic.utils.DistUtil;

/**
 * DCMLDA model is extension of LDA model, which can capture the burstiness of the
 * topic. For instance, the word that already appears in certain document, is likely to 
 * be appeared again in this document. 
 * 
 * For each document in LDA, we draw words from global topic-word distribution
 * \phi, but in DCMLDA, we draw words from document specific topic word distribution.
 * 
 * The model implements the collapsed gibbs sampler for inference, which is very similar
 * to LDA model, except that we draw word from document-specific distributions. However,
 * the most big difference in terms of inference is that we need to learn the hyperparameters
 * for DCMLDA model explicitly. The topic-word distribution prior \beta in DCMLDA actually
 * plays the similar role as \phi (global topic word distribution) in LDA. 
 * 
 * In order to learn these priors, we are using fixed point iteration method. Basically, 
 * we first calculate the likelihood for all the data given priors, and try to 
 * maximize that function using iterative manner. For more information about inference
 * part, please look at the report that comes with this code. 
 * 
 * @author wenzhe   nadalwz1115@hotmail.com
 *
 */
public class DCMLDAModel {
    // max iteration needed for updating alpha priors. 
    public final static int MAX_ALPHA_ITERATIONS = 30;
    // max iteration needed for updating beta priors
    public final static int MAX_BETA_ITERATIONS = 30;
    
    private int D;      // number of documents in the data set
    private int V;      // vocabulary size
    private int K;      // number of topics
    
    /**
     * Dirichlet hyperparameters. In DCMLDA model, these hyperparameters are very 
     * important. Thus, we need to estimate those explicitly. 
     */
    private double[] alpha;   // size K
    private double alphaSum;  // size 1
    private double[][] beta;  // size K * V
    private double[] betaSum; // size K
    
    /**
     * Parameters we need to estimate from the data. 
     */
    private float [][] theta;   // document-topic distributions, size D x K
    private float [][][] phi;   // doc-topic-word distributions, size D x K x V 
                                // in DCMLDA model, we keep each topic-word distribution for each document
    
    private int[][] z; // latent variable, topic assignments for each word. D * document.size()
    
    /**
     * Counting variables. Those variables are sufficient statistics for latent variable z
     */
    private short [][][] nDocTopicWord; // nDocTopicWord[i][k][v]: number of token v, that are in document i, and assigned to topic k, size D*K*V
    private int [][] nDocTopic;         // nDocTopic[i][k]: number of words in document i that assigned to topic k, size D x K
    private int [] nWordsInDoc;         // nWordsInDoc[i]: total number of words in document i, size D
    
    private DataSet dataset;  // input data set for learning.
     
    // initialize parameters. 
    public void init(DCMLDAOptions options, DataSet dataset){
        K = options.K;
        this.dataset = dataset;
        D = dataset.getDocumentCount();
        V = dataset.getVocabulary().getVocabularySize();
        
        // initialize hyperparameters using default values. 
        alpha = new double[K];
        beta = new double[K][V];
        betaSum = new double[K];
        for (int k=0; k < K; k++){
            alpha[k] = options.alpha;
            for (int v = 0; v < V; v++){
                beta[k][v] = options.beta;
            }
            betaSum[k] = options.beta * V;
        }
        alphaSum = options.alpha * K;
        
        theta = new float[D][K];
        phi = new float[D][K][V];
        
        // allocate memory for counting variables. 
        nDocTopicWord = new short[D][K][V];
        nDocTopic = new int[D][K];
        nWordsInDoc = new int[D];
        
        // initialize latent variable and counting variables. 
        z = new int[D][];
        for (int i = 0; i < D; i++){
            Document d = dataset.getDocument(i);
            int numTerms = d.getNumOfTokens();
            z[i] = new int[numTerms];
            for (int j = 0; j < numTerms; j++){
                int randTopic = (int)(Math.random() * K);
                z[i][j] = randTopic;
                
                nDocTopicWord[i][randTopic][d.getToken(j)]++;
                nDocTopic[i][randTopic]++;
            }
            nWordsInDoc[i] = numTerms;
        }   
    }
    
    // this will run one iteration of collapsed gibbs sampling.
    public void runGibbsSampler(){
        for (int i = 0; i < D; i++){
            Document d = dataset.getDocument(i);
            for (int j = 0; j < d.getNumOfTokens(); j++){
                // random sample z[i][j] 
                int newTopic = sampleNewTopic(i,j);
                z[i][j] = newTopic;
            }
        }
    }
    
    /**
     * Sample topic for jth word in ith document. 
     */
    private int sampleNewTopic(int i,  int j){
        Document d = dataset.getDocument(i);
        // remove current word..
        int oldTopic = z[i][j];
        nDocTopicWord[i][oldTopic][d.getToken(j)]--;
        nDocTopic[i][oldTopic]--;
        nWordsInDoc[i]--;
        
        // compute p(z[i][j]|*)
        double[] p = new double[K];
        for (int k = 0; k < K; k++){
            p[k] = ((alpha[k] + nDocTopic[i][k])/(alphaSum + nWordsInDoc[i])) 
                    * ((beta[k][d.getToken(j)]+nDocTopicWord[i][k][d.getToken(j)])/(betaSum[k]+nDocTopic[i][k]));
        }
        
        // sample the topic topic from the distribution p[j].
        int newTopic = DistUtil.sampleFromMultinomial(p);
        
        // add current word 
        nDocTopicWord[i][newTopic][d.getToken(j)]++;
        nDocTopic[i][newTopic]++;
        nWordsInDoc[i]++;
        
        return newTopic;
    }
    
    public void updateHyperparameters(int iteration){
        updateAlpha(iteration);
        updateBeta(iteration);
    }
    
    /**
     * update alpha until it is converged or exceed the maximum iterations. 
     * @param iteration     the current iteration
     */
    protected void updateAlpha(int iteration) {
        int currIteration = 0;
        double[] previousAlpha = new double[K];
        boolean b;
        do {
            System.arraycopy(alpha, 0, previousAlpha, 0, K);
            updateAlpha0(iteration);
            currIteration++;
            b = ConvergeUtil.arrayConverged(alpha, previousAlpha, Math.pow(10, -6));
        } while (!b && currIteration < MAX_ALPHA_ITERATIONS);
    }
    
    
    /**
     * update alpha vector, using fixed point iteration. 
     * Please see report for reference.
     * @param iteration
     */
    private void updateAlpha0(int iteration){
        double dividend = 0;
        for (int i = 0; i < D; i++){
            for (int j = 0; j < nWordsInDoc[i]; j++){
                dividend += 1/(j + alphaSum);
            }
        }
        
        for (int k = 0; k < K; k++){
            double divisor = 0;
            
            for (int i = 0; i < D; i++){
                for (int j = 0; j < nDocTopic[i][k]; j++){
                    divisor += 1/(j + alpha[k]);
                }
            }
            
            double newAlpha = Math.exp(Math.log(alpha[k]) + Math.log(divisor)
                    - Math.log(dividend));
            
            if (Double.isNaN(newAlpha) || Double.isInfinite(newAlpha)
                    || newAlpha < 0)
                newAlpha = 0;

            newAlpha = (alpha[k] * (iteration - 1) + newAlpha)
                    / iteration;

            alphaSum -= alpha[k];
            alpha[k] = newAlpha;
            alphaSum += alpha[k];
        }
    }
    
    
    /**
     * Update the beta Dirichlet parameter, topic-word distributions, until
     * it is converged or iteration exceed the maximum allowed one. 
     */
    protected void updateBeta(int iteration) {
        int currIteration = 0;
        double[][] previousBeta = new double[K][V];

        boolean b;
        do {
            for (short k = 0; k < K; ++k) {
                System.arraycopy(beta[k], 0, previousBeta[k], 0, V);
                System.out.println("update beta" + k);
                updateBeta0(k, iteration);
            }
            currIteration++;
            b = ConvergeUtil.matrixConverged(beta, previousBeta, Math.pow(10, -8));
        } while (!b && currIteration < MAX_BETA_ITERATIONS);
    }
    
    /**
     * Update beta vector using fixed point iteration. For derivation, refer 
     * to the report. 
     * @param k              the kth beta vector
     * @param iteration      the current iteration
     */
    protected void updateBeta0(short k, int iteration) {
        double dividend = 0;
        double divisor = 0; 
        
        for (int i = 0; i < D; i++)
            for (int j = 0; j < nDocTopic[i][k]; j++)
                dividend += 1.0 / (j + betaSum[k]);
        
        for (int v = 0; v < V; v++){
            divisor = 0;
            for (int i = 0; i < D; i++){
                for (int j = 0; j < nDocTopicWord[i][k][v]; j++){
                    divisor += 1.0/(j + beta[k][v]);
                }
            }
            
            double newBeta_k = Math.exp(Math.log(beta[k][v]) + Math.log(divisor)
                    - Math.log(dividend));

            if (Double.isNaN(newBeta_k) || Double.isInfinite(newBeta_k)
                    || newBeta_k < 0)
                newBeta_k = 0;

            newBeta_k = (beta[k][v] * (iteration - 1) + newBeta_k)
                    / iteration;

            betaSum[k] -= beta[k][v];
            beta[k][v] = newBeta_k;
            betaSum[k] += beta[k][v];
        }
    }
    
    public void updateParamters(){
        // update theta
        for (int i = 0; i < D; i++){
            for (int k = 0; k < K; k++){
                theta[i][k] = (float) ((alpha[k] + nDocTopic[i][k]) / (alphaSum + nWordsInDoc[i]));
            }
        }
        
        // update phi
        for (int i = 0; i < D; i++){
            for (int k = 0; k < K; k++){
                for (int v = 0; v < V; v++){
                    phi[i][k][v] = (float) ((beta[k][v] + nDocTopicWord[i][k][v]) / (betaSum[k] + nDocTopic[i][k])); 
                }
            }
        }
        
    }
    
    public float[][] getTopicDistribution(){
        return theta;
    }
    
    public float[][][] getDocTopicWordDistribution(){
        return phi;
    }
    
    public double[] getAlpha(){
        return alpha;
    }
    
    public double[][] getBeta(){
        return beta;
    }
}
