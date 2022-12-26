# Project: Sentiment Analysis with NLP
Sentiment Analysis on Movies Reviews

Basically a sentiment analysis problem consists of a classification problem, where the possible output labels are: `positive` and `negative`. Which indicates, if the review of a movie speaks positively or negatively. 


Regarding the data, we are going to download the data from [AI Stanford Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

### These are the objectives of the project:

* Read data that is not in a traditional format.

* Put together a set of preprocessing functions that we can use later on any NLP or related problems.

* Vectorize the data in order to apply a machine learning model to it: using BoW or TF-IDF.

* Train a sentiment analysis model that allows us to detect positive and negative opinions in movie reviews.

### ML models:

1. We train a word embedding from scratch, using:

    1.1 Random Forest classifier
    
    1.2 LightGBM classifier

   We use the two methods for vectorizing:

    * IF-TDF

    * BoW

    Besides, we use three methods to normalize the corpus:

    * Lemmatization

    * Stemm

    * No Lemma or Stemm (just simple normalization)


    So wehave 6 combinations, which we compare in a plot to evaluate the results    
    
    ![](https://github.com/DaDeCar/Sentiment-Analysis-with-NLP/blob/d9b025f455097b948fc06cfec94eb8afac89b71c/images/random_forest_roc_Auc%C3%A7.jpg)
    
    
    
    
    
    
    
2. We train a word embedding from scratch, using pre-trained models:

    2.1 Wikipedia
    
    2.2 Twitter
