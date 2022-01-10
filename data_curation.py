
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# IMDB Movie Reviews Dataset
imdb, info = tfds.load('imdb_reviews', with_info=True, split='train+test')
imdb= tfds.as_dataframe(imdb,info)
imdb['text'] = imdb['text'].str.decode('utf-8') # decode from bytestring
imdb['website'] = 'imdb' # keeping track of the website for later use
imdb = imdb[['website','text','label']]
imdb.to_csv('imdb_proc.csv')
# Yelp Restaurant Reviews Dataset
yelp, info = tfds.load('yelp_polarity_reviews', with_info=True, split='train+test')
yelp= tfds.as_dataframe(yelp,info)
yelp['text'] = yelp['text'].str.decode('utf-8')
yelp['website'] = 'yelp'
yelp = yelp[['website','text','label']]
yelp.to_csv('yelp.csv')
# Amazon Product Reviews Dataset
# load 4000000 amazon reviews from huggingface 
amazon_data = load_dataset("amazon_polarity") 
amazon_data['train'].to_csv('amazon_train.csv')
amazon_data['test'].to_csv('amazon_test.csv')
# load train test data into pandas df and combine to make dataset
amazon_train = pd.read_csv('amazon_train.csv')
amazon_test = pd.read_csv('amazon_test.csv')
amazon = pd.concat([amazon_train,amazon_test],ignore_index=True)
# drop unneccesary columns and rename to fit other datasets
amazon.drop(['title','Unnamed: 0'],axis=1,inplace=True)
amazon.rename(columns={'content': 'text'},inplace=True)
amazon['website'] = 'amazon'
amazon = amazon[['website','text','label']]
amazon.to_csv('amazon.csv')








