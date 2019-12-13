import re
import csv
import gensim
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

#set stopwords
stop_words = nltk.corpus.stopwords.words('english')
custom = ['?','(', ')', '.', '[', ']','!', '...',
        ';', "`", "'", '"',',', ':', '*', '~' , '/', '//', '\\']
stop_words.extend(custom)

#clean dataset by removing twitter mentions, links, and emojis
#set all letters to lowercase
#remove stopwords, tokenize, and lemmatize tweets
def clean_data(raw_data):

    remove_mentions = re.sub(r'@[A-Za-z0-9]+', '', raw_data)
    remove_links = re.sub('https?://[A-Za-z0-9./]+', '', remove_mentions, flags=re.MULTILINE)
    remove_bitly_links = re.sub(r'bit.ly/\S+', '', remove_links)
    remove_non_ascii = re.sub(r'[^\x00-\x7F]+',' ', remove_bitly_links)
    set_lowercase = remove_non_ascii.lower()
    tokenized = TweetTokenizer().tokenize(set_lowercase)
    remove_stopwords = [words for words in tokenized if not words in stop_words]
    lemmatized = [WordNetLemmatizer().lemmatize(word) for word in remove_stopwords]

    return lemmatized


def main():

    #initialize variables and lists
    count = 0
    tweet_list =[]
    list_list = []

    #access csv file and read out data
    csv_file = open('tweet_data.csv')
    data = csv.reader(csv_file)

    #gather all tweets and apply clean)data() function
    for row in data:
        string_row = str(row).strip('[]')
        tweets = clean_data(string_row)

        for word in tweets:
            tweet_list.append(word)

    #place into one giant list
    list_list.append(tweet_list)

    #place all text into a bag of words model
    dictionary = gensim.corpora.Dictionary(list_list)

    #run dictionary into a corpus to get frequency
    bow_corpus = [dictionary.doc2bow(i) for i in list_list]


    #train model
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=4, id2word=dictionary, passes=10, workers=2)

    for idx, topic in lda_model.print_topics(-1):
        print("Topic: {} \nWords: {}".format(idx, topic))
        print('\n')

if __name__ == "__main__":
    main()
