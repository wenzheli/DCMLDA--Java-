package ml.topic.dcmlda;

import java.util.Arrays;

import math.DifferentiableFunction;
import math.LBFGSMinimizer;
import ml.topic.common.data.DataSet;
import ml.topic.common.data.Document;
import ml.topic.utils.DistUtil;

public class DCMLDAModel {
    
    private int D;  // number of documents in the data set
    private int V;  // vocabulary size
    
    
    private int K;  // # of topic
    
    /**
     * Dirichlet hyperparameters. In DCMLDA model, these hyperparameters are very 
     * important. Thus, we need to estimate those explicitly. 
     */
    private double[] alpha;   // size K
    private double[][] beta;  // size K * V
    private double alphaSum;  
    private double[] betaSum; // size K
    
    /**
     * Parameters we need to estimate from the data. 
     */
    private float [][] theta;   // document-topic distributions, size D x K
    private float [][][] phi;   // doc-topic-word distributions, size D x K x V 
                                 // in DCMLDA model, we keep each topic-word distribution for each document
    
    private int[][] z; // latent variable, topic assignments for each word. D * document.size()
    
    private short [][][] nDocTopicWords; // nTopicWords[i][j]: # of instances of word/term j assigned to topic i, size K*V
    private int [][] nDocTopic;   // nDocTopic[i][j]: # of words in document i that assigned to topic j, size D x K
    private int [] nWordsSum;     // nWordsSum[i]: total number of words in document i, size D
    
    private DataSet dataset;
     
    // initialize parameters. 
    public void init(Options options, DataSet dataset){
        this.K = options.K;
        this.dataset = dataset;
        this.D = dataset.getDocumentCount();
        this.V = dataset.getVocabulary().getVocabularySize();
        
        // initialize hyperparameters using default values. 
        alpha = new double[K];
        beta = new double[K][V];
        betaSum = new double[K];
        for (int k=0; k<K; k++){
            alpha[k] = options.alpha;
            for (int v = 0; v < V; v++){
                beta[k][v] = options.beta;
            }
            betaSum[k] = options.beta*V;
        }
        alphaSum = options.alpha * K;
        
        // these parameters are sufficient statistics of latent variable Z. We only sample z instead
        theta = new float[D][K];
        phi = new float[D][K][V];
        
        // allocate memory for temporary variables
        nDocTopicWords = new short[D][K][V];
        nDocTopic = new int[D][K];
        nWordsSum = new int[D];
        
        // initialize latent variable - z
        z = new int[D][];
        for (int i = 0; i < D; i++){
            WDocument d = (WDocument) dataset.getDocument(i);
            int numTerms = d.getNumOfTokens();
            z[i] = new int[numTerms];
            for (int j = 0; j < numTerms; j++){
                int randTopic = (int)(Math.random() * K);
                z[i][j] = randTopic;
                
                nDocTopicWords[i][randTopic][d.getToken(j)]++;
                nDocTopic[i][randTopic]++;
            }
            nWordsSum[i] = numTerms;
        }   
    }
    
    // this will run one iteration of collapsed gibbs sampling.
    public void runGibbsSampler(){
        for (int i = 0; i < D; i++){
            WDocument d = (WDocument) dataset.getDocument(i);
            for (int j = 0; j < d.getNumOfTokens(); j++){
                // random sample z[i][j] 
                int newTopic = sampleNewTopic(i,j);
                z[i][j] = newTopic;
            }
        }
    }
    
    private int sampleNewTopic(int i,  int j){
        WDocument d = (WDocument) dataset.getDocument(i);
        int oldTopic = z[i][j];
        nDocTopicWords[i][oldTopic][d.getToken(j)]--;
        nDocTopic[i][oldTopic]--;
        nWordsSum[i]--;
        
        // compute p(z[i][j]|*)
        double[] p = new double[K];
        for (int k = 0; k < K; k++){
            p[k] = ((alpha[k] + nDocTopic[i][k])/(alphaSum + nWordsSum[i])) 
                    * ((beta[k][d.getToken(j)]+nDocTopicWords[i][k][d.getToken(j)])/(betaSum[k]+nDocTopic[i][k]));
        }
        
        // sample the topic topic from the distribution p[j].
        int newTopic = DistributionUtils.getSample(p);
        
        nDocTopicWords[i][newTopic][d.getToken(j)]++;
        nDocTopic[i][newTopic]++;
        nWordsSum[i]++;
        
        return newTopic;
    }
    
    public void updateAlphaLBFGS(){
        LBFGSMinimizer minimizer = new LBFGSMinimizer();
        double[] init = new double[K];
        Arrays.fill(init, 1);
        double[] ans = minimizer.minimize(new DifferentiableFunction() {
         
            @Override
            public double valueAt(double[] alpha) {
                   double sum = 0;
                   double num1 = 0;
                   for (int k = 0; k < K; k++){
                       num1 -= logGamma(alpha[k]);
                   }
                   sum += num1 * D;
                   sum += D * logGamma(alphaSum);
                   
                   for (int i = 0;i < D;i++){
                       for (int k = 0;k < K;k++){
                           sum += logGamma(alpha[k]+nDocTopic[i][k]);
                       }
                   }
                   
                   for (int i = 0;i < D;i++){
                       sum -= logGamma(alphaSum + nWordsSum[i]); 
                   }
                   
                   return -1 * sum;  
            }

            @Override
            public int dimension() {
                    return K;
            }

            @Override
            public double[] derivativeAt(double[] alpha) {
                double[] derivative = new double[K];
                double constant = 0;
                constant += D * digamma(alphaSum);
                for (int i = 0;i < D;i++){
                    constant -= digamma(alphaSum + nWordsSum[i]);
                }
                
                for (int k = 0; k < K; k++){
                    for (int i = 0; i < D; i++){
                        derivative[k] += digamma(alpha[k] + nDocTopic[i][k]);
                    }
                    derivative[k] -= D * digamma(alpha[k]);
                    derivative[k] += constant;
                }
                    
                return derivative;
            }
        }, init, 1e-4);
        
        alpha = ans;
        alphaSum = 0;
        for (int k = 0; k < K; k++){
            alphaSum += alpha[k];
        }
    }
    
    public void updateHyperparameters(){
        updateAlphaLBFGS();
        System.out.println("finishing updating alpha..");
        for (int k = 0; k < K; k++){
            System.out.println("updating beta for" + k + "th beta");
            updateBetaLBFGS(k);
        }
        //updateAlpha();
        //updateBeta();
    }
    
    public void updateBetaLBFGS(final int k){
        LBFGSMinimizer minimizer = new LBFGSMinimizer();
        double[] init = new double[V];
        Arrays.fill(init, 0.01);
        double[] ans = minimizer.minimize(new DifferentiableFunction() {  
            @Override
            public double valueAt(double[] beta1) {
                double sum = 0;
                for (int v = 0; v < V; v++){
                    sum -= logGamma(beta1[v]);
                }
                sum = sum * D;
                sum += D * logGamma(betaSum[k]);
                
                for (int i = 0; i < D; i++){
                    for (int v = 0; v < V; v++){
                        sum += logGamma(nDocTopicWords[i][k][v] + beta1[v]);
                    }
                    sum -= logGamma(nDocTopic[i][k] + betaSum[k]);
                }
                
                return - 1 * sum;
            }

            @Override
            public int dimension() {
                    return V;
            }

            @Override
            public double[] derivativeAt(double[] beta1) {
                double[] derivative = new double[V];
                double constant = 0;
                constant -= D * digamma(betaSum[k]);
                for (int i = 0;i < D;i++){
                    constant -= digamma(betaSum[k] + nDocTopic[i][k]);
                }
                
                for (int v = 0; v < V; v++){
                    for (int i = 0;i < D;i++){
                        derivative[v] += digamma(beta1[v] + nDocTopicWords[i][k][v]);
                    }
                    derivative[v] -= D * digamma(beta1[v]);
                    derivative[v] += constant;
                }
                
                return derivative;
            }
        }, init, 1e-4);
        
        beta[k] = ans;
        betaSum[k] = 0;
        for (int v = 0; v < V; v++){
            betaSum[k] += beta[k][v];
        }
    }
    
    private void updateAlpha(){
        int currIteration = 0;
        int maxIteration = 30;
        double[] previousAlpha = new double[K];
        double dividend = 0;
        double divisor = 0;
        do {
            if (currIteration > maxIteration)
                break;
            System.arraycopy(alpha, 0, previousAlpha, 0, K);

            dividend = -D * digamma(alphaSum);

            for (int i = 0; i < D; i++)
                dividend += digamma(nWordsSum[i] + alphaSum);

            for (int k = 0; k < K; k++) {
                divisor = -D * digamma(alpha[k]);

                for (int i = 0; i < D; i++)
                    divisor += digamma(nDocTopic[i][k]
                            + alpha[k]);
                double newAlpha = alpha[k] * divisor / dividend;;
                if (Double.isNaN(newAlpha) || Double.isInfinite(newAlpha)
                        || newAlpha < 0)
                    newAlpha = 0;
                
                alphaSum -= alpha[k];
                alpha[k] = newAlpha;
                alphaSum += alpha[k];
            }
            currIteration++;

        } while (!arrayConverged(alpha, previousAlpha, Math.pow(
                10, -6)));
    }
    

    private void updateBeta() {
        System.out.println("updating beta...");
        double[][] previousBeta = new double[D][V];
        
        boolean b;
        int currIteration = 0;
        int maxIteration = 30;
        do {
            System.out.println("current iteration: " + currIteration);
            for (int k = 0; k < K; ++k) {
                System.arraycopy(beta[k], 0, previousBeta[k], 0, V);
                updateBeta0(k);
            }
            currIteration++;
            b = matrixConverged(beta, previousBeta, Math.pow(10, -8));
        } while (!b && currIteration < maxIteration);
    }
    
    private void updateBeta0(int k){
        double dividend = 0;
        double divisor = 0;
        
        dividend = -D * digamma(betaSum[k]);

        for (int i = 0; i < D; i++)
            dividend += digamma(nDocTopic[i][k] + betaSum[k]);

        for (int v = 0; v < V; v++) {
            divisor = -D * digamma(beta[k][v]);

            for (int i = 0; i < D; i++){
                divisor += digamma(nDocTopicWords[i][k][v]
                        + beta[k][v]);
            }
            double newBeta = beta[k][v] * divisor / dividend;;
            if (Double.isNaN(newBeta) || Double.isInfinite(newBeta)
                    || newBeta < 0)
                newBeta = 0;
            betaSum[k] -= beta[k][v];
            beta[k][v] = newBeta;
            betaSum[k] += beta[k][v];
        }
    }
    
       
    public void updateParamters(){
        // update theta
        for (int i = 0; i < D; i++){
            for (int k = 0; k < K; k++){
                theta[i][k] = (float) ((alpha[k] + nDocTopic[i][k]) / (alphaSum + nWordsSum[i]));
            }
        }
        
        // update phi
        for (int i = 0; i < D; i++){
            for (int k = 0; k < K; k++){
                for (int v = 0; v < V; v++){
                    phi[i][k][v] = (float) ((beta[k][v] + nDocTopicWords[i][k][v]) / (betaSum[k] + nDocTopic[i][k])); 
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
