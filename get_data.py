import csv
import sys
import tweepy
from cred import *

#get twitter api using credentials in cred.py
def get_api():

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    try:
        api.verify_credentials()
        print('successfully created twitter API')
        return api

    except Exception as e:
        print('Error creating API')
        raise e


#get tweets from twitter users using a list of their screen names
def get_tweets(twitter_api, username):

    output = 'tweet_data.csv'
    number_of_tweets = 10000
    tweet_list = []

    #make sure not to include retweets
    for tweet in tweepy.Cursor(twitter_api.user_timeline, tweet_mode='extended', include_rts=False, screen_name=username).items(number_of_tweets):
        tweet_list.append([tweet.full_text])

    #write tweets onto a csv file
    with open(output, 'a') as file:

        writer = csv.writer(file, delimiter=',')
        writer.writerows(tweet_list)


def main():

    #supply users here
    
    usernames=[]

    #get api
    api = get_api()
    try:
        #get tweets
        for user in usernames:
            print('Currently on ' + user)
            get_tweets(api, user)
    except:
        pass
        print('error with: ' + user + ' moving to next user...')
        

if __name__ == "__main__":

    main()


    '''
    usernames=['demishassabis',
                'goodfellow_ian',
                'JeffDean',
                'karpathy',
                'ch402',
                'iamtrask',
                'trentmc0',
                'gdb',
                'NandoDF',
                'ilyasut',
                'AndrewYNg',
                'MituK',
                'msbernst',
                'jeffbigham',
                'GoogleAI',
                'TensorFlow',
                'IBMResearch',
                'MSFTResearch',
                'srush_nlp',
                'amyxzh',
                'AngelBassa',
                'richardyoung00',
                'sophiebushwick',
                'seanmcarroll',
                'EricTopol',
                'maletsabisam',
                'DimaKrotov',
                'AnnaPaolaCarri',
                'TeddySeyed',
                'IntelAI',
                'datasociety',
                'facebookai',
                'KDziugaite',
                'LRieswijk',
                'FryRsquared',
                'kaifulee',
                'DeepMind_Health',
                'OriolVinyalsML',
                'blaiseaguera',
                'drfeifei',
                'Dominic1King',
                'ShaneLegg',
                'weissg1234',
                'sciam',
                'svlevine',
                'sanakb',
                'tanmingxing',
                'quocleix',
                'hardmaru',
                'PyTorch',
                'an_open_mind',
                '_inesmontani',
                'jsusskin',
                'amuellerml',
                'chelseabfinn',
                'markus_with_k',
                'julien_c',
                'erichorvitz',
                'fchollet',
                'OpenAI',
                'gpapamak',
                'distillpub',
                'stanfordnlp',
                'BaiduResearch',
                'NvidiaAI',
                'StanfordAILab',
                'StanfordHAI',
                'Deep_AI',
                'arxiv_org',
                'red_abebe',
                'rapidsai',
                'NVIDIAAIDev',
                'yaringal',
                'dustinvtran',
                'roydanroy',
                'andrewgwils',
                'santoroAI',
                'DavidDuvenaud',
                'ericjang11',
                'allen_ai',
                '3blue1brown',
                'diff_eq',
                'MedVocab',
                'jabrils_',
                'TheSpaceGal',
                'poolio',
                'SuryaGanguli',
                'QuantaMagazine',
                'lavanyaai',
                'weights_biases',
                '_beenkim',
                'SimoneGiertz'
                '''
