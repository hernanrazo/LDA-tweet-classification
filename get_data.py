import csv
import sys
import tweepy
from cred import *


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




def get_tweets(twitter_api, username):

    number_of_tweets = 100
    tweet_list = []

    for tweet in tweepy.Cursor(twitter_api.user_timeline, tweet_mode='extended', include_rts=False, screen_name=username).items(number_of_tweets):
        tweet_list.append([tweet.full_text])

    output = 'tweet_data.csv'

    with open(output, 'a') as file:

        writer = csv.writer(file, delimiter=',')
        writer.writerows(tweet_list)


def main():


    usernames=[]

    api = get_api()

    for user in usernames:

        print('Currently on ' + user)
        get_tweets(api, user)




if __name__ == "__main__":

    main()
