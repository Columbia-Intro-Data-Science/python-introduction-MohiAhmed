import tweepy
import re
import re
from os import path
from wordcloud import WordCloud
import requests
import time
from datetime import datetime
import pandas as pd
import json
from twitter_common import *
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
trends1 = api.trends_place({NUMBER_HERE})

def get_trending_hashtags():
    # OAuth process, using the keys and tokens
    # trends1 is a list with only one element in it, which is a
    # dict which we'll put in data.
    data = trends1[0]
    # grab the trends
    trends = data['trends']
    # grab the name from each trend
    names = [trend['name'] for trend in trends]
    # put all the names together with a ' ' separating them
    return names


top_hashtags = get_trending_hashtags()

def clean_tweets(final_text):
    final_text = re.sub('"','',final_text)
    final_text = re.sub("'","",final_text)
    final_text = re.sub("#","",final_text)
    final_text = re.sub("@","",final_text)
    return final_text

def get_top_tweets(search_text):
    search_number = 20
    search_result = api.search(search_text, rpp=search_number,lang="en",include_entities=False)

    final_text = ""
    for i in search_result:
        final_text = final_text + i.text
    final_text = clean_tweets(final_text)

    return final_text

def append_historical_twitter(df_trending_now):
	df_historical = pd.read_csv(PATH + 'data/twitter_proccessed.tsv',sep='\t')
	df_trending_now = df_trending_now.fillna(0).groupby('timestamp')[topic_cols].mean()
	df_trending_now['timestamp']=df_trending_now.index
	df_updated = df_trending_now.append(df_historical)
	df_updated['timestamp_hour'] = df_updated['timestamp'].values.astype('<M8[h]')
	df_updated=df_updated.reset_index(drop=True)

	return df_updated


if __name__ == "__main__":
	current_time=str(datetime.now())

	print 'Current time is ', current_time

	# Getting top hashtags
	try:
		top_hashtags = get_trending_hashtags()
	except:
		print 'Failed to load hashtags'
		exit();
	print top_hashtags
	# Create dataframe to store trending topics
	df_trending_now = pd.DataFrame(index=range(num_tweets),columns=twitter_cols)
	df_trending_now['timestamp']=current_time
	wordcloud_text = ""
	# Cycle through tweets
	for i in range(num_tweets):
	    # Get top tweets from given hashtag
	    tweet_data=get_top_tweets(top_hashtags[i])

	    print (tweet_data)
