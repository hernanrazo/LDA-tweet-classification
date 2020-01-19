import re
import pickle
import numpy as np
import pandas as pd
import gensim
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


#set stopwords
stop_words = nltk.corpus.stopwords.words('english')
custom = ['?','(', ')', '.', '[', ']','!', '...', '-', '@', '->','https',
        ';', "`", "'", '"',',', ':', '*', '~' , '/', '//', '\\', '&', 'n', ':\\']
stop_words.extend(custom)

#convert words to bigrams
def get_bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count=bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod


#get raw string and remove mentions, links, and emojis. Also set all letters to lowercase and filter stopwords
def clean_tweet(raw_tweet):
    remove_mentions = re.sub(r'@[A-Za-z0-9]+', '', raw_tweet)
    remove_links = re.sub('https?://[A-Za-z0-9./]+', '', remove_mentions, flags=re.MULTILINE)
    remove_bitly_links = re.sub(r'bit.ly/\S+', '', remove_links)
    remove_non_ascii = re.sub(r'[^\x00-\x7F]+', '', remove_bitly_links)
    set_lowercase = remove_non_ascii.lower()
    token = word_tokenize(set_lowercase)
    filtered = [words for words in token if not words in stop_words]
    return filtered


def main():
    #retrieve saved pickles and model from training
    model = pd.read_pickle('saved_pickles_models/lda_model.model')
    train_id2word = pd.read_pickle('saved_pickles_models/train_id2word.pkl')
    huber_classifier = pd.read_pickle('saved_pickles_models/huber_classifier.pkl')

    #example tweets
    test1 = 'my machine learning model uses k-means clustering and the data was cleaned using spacy. made by @hrazo7. can be found here https://github.com/hrazo7    🏁'
    test2 = 'this ice cream tastes very good. I bought it at the farmers market. The ingredients looked very fresh.......'
    test3 = 'Open Syllabus Project https://opensyllabus.org Open Syllabus is a non-profit organization that maps the curriculum of higher education. Database of / stats from 7M class syllabi 👏'
    test4 = 'Thanks!'
    test5 = 'this conference was a complete waste of time. The lines were long, the swag was minimal, and I didnt even get to meet all the people I indirectly talk shit about on twitter smh'

    #gather all test tweets into a list and iterate through it later
    tweet_list = [test1, test2, test3, test4, test5]
    test_vector_list = []
    scaler = StandardScaler()

    #iterate through each tweet in the tweet list. 
    #alternatively, remove the for loop and pass a single tweet string to the clean_tweet() function call
    for tweet in tweet_list:
        #clean tweet, get bigram, and get corpus
        cleaned_tweet = clean_tweet(tweet)
        bigram = get_bigrams(cleaned_tweet)
        full_bigram = bigram[cleaned_tweet]
        test_corpus = train_id2word.doc2bow(full_bigram)

        #use saved model and modified huber pickle to predict the topic
        top_topics = model.get_document_topics(test_corpus, minimum_probability=0.0)
        topic_vector = [top_topics[i][1] for i in range(15)]
        test_vector_list.append(topic_vector)

        #take topic vectors and use them to make a prediction
        x = np.array(test_vector_list)
        x_fit = scaler.fit_transform(x)
        prediction = huber_classifier.predict(x_fit)
        score = top_topics[prediction[0]][1]

        #only grab tweets within a high enough score
        if score > 0.60:
            print('Found relevant tweet:')
            print(tweet)
            print('\n')

        else:
            print('Not relevant tweet:')
            print(tweet)
            print('\n')

if __name__ == "__main__":
    main()
