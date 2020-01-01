import pickle
import csv
import pandas as pd

feature_vector = []
data = pd.read_csv('tweet_data.csv')
bigram = pd.read_pickle('train_bigram.pkl')
id2word = pd.read_pickle('train_id2word.pkl')
corpus = pd.read_pickle('train_corpus.pkl')
model = pd.read_pickle('lda_model.model')

for i in range(len(data)):
    top_topics = model.get_document_topics(corpus[i], minimum_probability=0.0)
    topic_vector = [top_topics[i][1] for i in range(15)]
    feature_vector.extend(topic_vector)
print(feature_vector[2])
'''
    topic_vector.extend([len(data.iloc[i].text)])
    vector.append(topic_vector)
print(vector[2])
'''
