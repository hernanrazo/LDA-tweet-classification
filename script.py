import re
import csv
import pandas as pd
import gensim
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

stemmer = SnowballStemmer('english')
stop_words = nltk.corpus.stopwords.words('english')
custom = ['?','(', ')', '.', '[', ']','!', '...',
        ';',"`","'",'"',',', ':', '~' , '/', '//', '\\'] 
stop_words.extend(custom)


def clean_data(raw_data):


    remove_mentions = re.sub(r'@[A-Za-z0-9]+', '', raw_data)
    remove_links = re.sub('https?://[A-Za-z0-9./]+', '', remove_mentions, flags=re.MULTILINE)
    remove_bitly_links = re.sub(r'bit.ly/\S+', '', remove_links)
    set_lowercase = remove_bitly_links.lower()
    tokenized = TweetTokenizer().tokenize(set_lowercase)
    remove_stopwords = [words for words in tokenized if not words in stop_words]
    lemmatized = [WordNetLemmatizer().lemmatize(word) for word in remove_stopwords]

    return lemmatized


def main():

    f = open('tweet_data.csv')
    csv_f = csv.reader(f)
    
    for row in csv_f:

        string_row = str(row).strip('[]')
        tweets = clean_data(string_row)

        print(tweets)




if __name__ == "__main__":

    main()





    '''


stemmer = SnowballStemmer('english')
stop_words = nltk.corpus.stopwords.words('english')
custom = ['?','(', ')', '.', '[', ']','!', '...',
';',"`","'",'"',',' ] #"'s", "'ll", 'ca', "n't", "'m", "'re", "'ve"]
stop_words.extend(custom)


def clean_data(raw_data):

    remove_mentions = re.sub(r'@[A-Za-z0-9]+', '', raw_data)
    remove_links = re.sub('https?://[A-Za-z0-9./]+', '', remove_mentions, flags=re.MULTILINE)
    remove_bitly_links = re.sub(r'bit.ly/\S+', '', remove_links)
    set_lowercase = remove_bitly_links.lower()
    tokenized = TweetTokenizer().tokenize(set_lowercase)





    return tokenized


def main():

    df = pd.read_csv('tweet_data.csv')

    pd.set_option('display.max_colwidth', -1)
    print(df.head(10))
    print('==========================================================================')
    tweets = df[df.columns[0]].apply(lambda x: clean_data(x))

    print(tweets)




if __name__ == "__main__":

    main()
    '''
