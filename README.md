DCMLDA(Java version)
=============

## Introduction
DCMLDA is an extension of LDA model that used for capturing word burstiness, which means if a term is used once
in a document, then it is likely to be used again. There is classifical topic model like LDA, does not capture burstiness, 
while DCMLDA is specifically designed for capturing this phenomena. The implementation is based on Java, and should
work for any text corpus with very minimal modification (i.e text processing for your own corpus). 


## Then what are differences between LDA and DCMLDA? 
The figure below shows how two models differ each other. As we can see, the difference is for topic-word distribution \psi. 
In LDA model, each document draw words from global topic word distribution \psi, but in DCMLDA model, we each document 
draw words from document-specific topic-word distributions. 

The interesting thing here is, the prior \beta in DCMLDA plays very similar role as \psi in LDA model.  Thus, we cannot
simply use default values for \beta in DCMLDA model, but we need to learn the optimal prior, \alpha and \beta using 
inference techniques. 


### Generative Process of DCMLDA
``` ruby
for document d \in {1,...D} do
  draw topic distribution \theta_{d} from Dir(\alpha)
  for topic k \in {1,...K} do
    draw topic-word distribution \phi_{d,k} from Dir(\beta_{k})
  end for
  for each word in document d
    draw topic z_{d} from multinomial distribution \theta_{d}
    draw word w_{d,n} from multinomial distribution \phi_{d,z_{dn}}
  end for
end for
```

### Technical detail about DCMLDA model
All the technical details (derivations, etc) are included in the technical report - DCMLDA.pdf


## Quick Start
### Install the package 
``` 
git clone git@github.com:wenzheli/DCMLDA--Java-.git
```
### Try example: analyzing NIPS corpus
Run DCMLDAInference.java, no need to provide any arguments. 
### Try modifying the input parameters. 
- -nTopics: number of topics
- -alpha: default value for alpha prior
- -beta: default value for beta prior
- -burnin: number of iterations before sampling starts
- -samples: number of samples for estimating parameters
- -sampleWait: number of iterations for each sampling period
- -tWords: number of top words for each topic
- -file: input data file

## Implementation

### Parse the text and pre-processing
In general, for any given corpus, we need to create approprieate data structore for storing those text. 
In this java implementation, we simply uses java List<Integer> to represents each document, where this list
contains the sorted list of tokends as appears in the document. There is one common implementation only
store the occurance of each token, without regarding the order of those words (but this one does not work for
bi-gram or n-gram model, where we need to maintain the order of tokens)

After reading in all the text, we need to do some text pre-processing. 1. Stemming the words, and 2. Remove the 
stop words. For stemming, we used the porter stemmer http://tartarus.org/martin/PorterStemmer/java.txt.

Besides those two steps, we also did some text processing like remove any token that contains digit. (Using or not
depends on your case anyway..)

### Initialize the parameters 
By default, you can initialize alpha as 50/(# of topics), and beta as 0.01. But again, in DCMLDA model, you need
to explicitly learn these hyperparameters. 

### Train the model using collapsed gibbs sampler (burn-in period)
Collapsed gibbs sampler simply samples topic for each word in the document. Basically, it calculate the probability 
p(z_{j}|z_{-j},W,\alpha,\beta). The gibbs sampler process is slightly different from LDA model. 

### Estimate the hyperparameters. 
The main challenging part for DCMLDA is to learn hyperparameters, alpha and beta. In general, there are two ways
to learn these hyperparameters. 

- As shown in the original paper (cseweb.ucsd.edu/~elkan/TopicBurstiness.pdf), we can calculate the likelihood given
all the documents, and then maximize them by choosing optimal hyperparameters. This involes solving K+1 non-linear
equations. L-BFGS is the one can solve it for us. (http://en.wikipedia.org/wiki/Limited-memory_BFGS). For java version, 
you can check the code written by Dan Klein from UC Berkeley. 

- Another approach is to use some iterative approach for finding optimal values. One typical way is to use fixed point
iteration (http://en.wikipedia.org/wiki/Fixed-point_iteration). Actually, the code is using this method. For detailed
derivaition, refer to the report. 

### Perplexity score. 
In NLP, perplexity score is one of the important measure for goodness of fit. This simply calculate the maximum likelihood
for the testing documents. 

### Print the top words for each topic. 
Lastly, we are always curious to see top words per each topic. In LDA model, \phi is capturing this information while in
DCMLDA model, prior \beta contain this information. 


## Reference
- G. Doyle and C. Elkan. Accounting for Word Burstiness in Topic Models In Proceedings of the 26th International Conference on Machine Learning (ICML), July 2009
- Nicola Barbieri, Giuseppe Manco, Ettore Ritacco, Marco Carnuccio, Antonio Bevacqua. Probabilistic topic models for sequence data
Machine Learning October 2013, Volume 93, Issue 1, pp 5-29


## Have any problems for code? 
Send email to nadalwz1115@hotmail.com, appreciate your feedback. 
