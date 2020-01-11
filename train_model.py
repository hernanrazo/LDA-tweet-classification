import pickle
import csv
import logging
import warnings
import gensim
import nltk
from nltk.corpus import stopwords

#set stopwords
stop_words = nltk.corpus.stopwords.words('english')
custom = ['?','(', ')', '.', '[', ']','!', '...', '-', '@', '->','https',
        ';', "`", "'", '"',',', ':', '*', '~' , '/', '//', '\\', '&', 'n', ':\\']
stop_words.extend(custom)


#process csv content to a list
def csv_to_words(raw_data):
    for row in raw_data:
        yield(gensim.utils.simple_preprocess(str(row), deacc=True))


#remove stopwords
def remove_stopwords(processed_data):
    return [[words for words in gensim.utils.simple_preprocess(str(doc)) if not words in stop_words] for doc in processed_data]


#convert words into bigrams
def get_bigram(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count=bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod


def main():
    #set vars, logging, and open csv file
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    tweet_list = []
    print('Opening file...')
    csv_file = open('tweet_data.csv')
    data = csv.reader(csv_file)

    print('Cleaning data...')
    #gather all tweets and apply the clean_data() and get_bigram() functions
    processed_tweets = list(csv_to_words(data))
    cleaned_tweets = remove_stopwords(processed_tweets)
    bigrams= get_bigram(cleaned_tweets)
    bigram = [bigrams[tweet] for tweet in cleaned_tweets]
    id2word = gensim.corpora.Dictionary(bigram)
    id2word.compactify()
    corpus = [id2word.doc2bow(tweets) for tweets in bigram]

    print('Training model...')
    print('\n')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus, num_topics=10, id2word=id2word, chunksize=100, workers=2, passes=50, eval_every=1, per_word_topics=True)


    #print list of topics
    print('\n')
    print('Geting topics...')
    lda_model.print_topics(15, num_words=15)[:15]
    print('\n')


    #save everything
    print('Saving everything...')
    bigram_save = open('train_bigram.pkl', 'wb')
    pickle.dump(bigram, bigram_save)
    bigram_out.close()

    id2word_save = open('train_id2word.pkl', 'wb')
    pickle.dump(id2word, id2word_save)
    id2word_out.close()

    corpus_save = open('train_corpus.pkl', 'wb')
    pickle.dump(corpus, corpus_save)
    corpus_out.close()

    model_save = open('lda_model.model', 'wb')
    pickle.dump(lda_model, model_save)
    model_out.close()
    print('Done')

if __name__ == "__main__":
    main()
