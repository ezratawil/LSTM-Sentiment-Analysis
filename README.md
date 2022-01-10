# LSTM-Sentiment-Analysis
Predicting Sentiment Polarity of customer reviews using Recurrent Neural Networks with GloVe Word Embeddings 

# Goal
Train a Model to generalize sentiment classification of customer reviews from different categories

# Data
Data is comprised of Movie Reviews(IMDB),Yelp Reviews (Restaurants), and Amazon Reviews (Products). Data was sourced from HuggingFace Datasets library and TensorFlow Datasets library. Review polarity is labelled as positive or negative.

# Preprocessing
Data manpulation and cleaning of review text by removal of punctuation, casing, stopwords, rare words, frequent words as well as Tokenizing words and converting to padded sequences to prepare for model input

![review processing](https://www.researchgate.net/profile/Norah-Alghamdi-4/publication/334185442/figure/fig3/AS:776401590439936@1562119972322/Tokenization-for-Amazon-website-unbalanced-dataset-of-mobile-phone-reviews.ppm)

# Embedding Text - Tranfer Learning Using Stanford's GloVe (Global Vectors for Word Representation)
Word Embedding is an approach used to provide a dense vector representation of words that capture some context words about their own. These are improved versions of simple Bag of Words models like word counts and frequency counts. Vector Space representation of words provides a projection where words of similar semantics are clustered together within the space.

![Vector space embed](https://d1m75rqqgidzqn.cloudfront.net/wp-data/2020/07/03000751/we1.png)
# Model - Keras 
LSTM Recurrent Neural Network , Embedding layer Pre-Initialized with weight matrix created from Stanford's GloVe Word Embeddings. Model was trained using google colab with TPU hardware acceleration
