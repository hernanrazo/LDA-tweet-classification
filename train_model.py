import re
import pickle
import logging
import warnings
import pandas as pd
import gensim
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

#set stopword
stop_words = nltk.corpus.stopwords.words('english')
custom = ['?','(', ')', '.', '[', ']','!', '...', '-', '@', '->', 'https', 'http',
        ';', "`", "'", '"',',', '``', "''", ':', '*', '~' , '/', '//', '\\', '&', 'n', ':\\']
stop_words.extend(custom)


def clean_status(data):
    remove_mentions = re.sub(r'@[A-Za-z0-9]+', '', data)
    remove_links = re.sub('https?://[A-Za-z0-9./]+', '', remove_mentions, flags=re.MULTILINE)
    remove_bitly_links = re.sub(r'bit.ly/\S+', '', remove_links)
    remove_non_ascii = re.sub(r'[^\x00-\x7F]+', '', remove_bitly_links)
    set_lowercase = remove_non_ascii.lower()
    token = word_tokenize(set_lowercase)
    filtered = [words for words in token if not words in stop_words]
    return filtered


#convert words into bigrams
def get_bigram(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count=bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod


def main():
    #set vars, logging, and open csv file
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print('Opening file...')
    train_data = pd.read_csv('data/train_data.csv', encoding='ISO-8859-1')
    status_list = []

    for row in train_data['tweets']:
        cleaned_status = clean_status(row)
        status_list.append(cleaned_status)
    bigrams = get_bigram(status_list)
    bigram = [bigrams[entry] for entry in status_list]
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
    lda_model.print_topics(10, num_words=10)[:10]
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
