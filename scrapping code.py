# -*- coding: utf-8 -*-
# I find snscrape so efficient in scrapping Twitter. The only downside is the scrapping speed, and it only works under the environment of python 3.8
# itertools
import itertools
import pandas as pd
import snscrape.modules.twitter as sntwitter
# I tried to scrap 1000000 tweets which took me nearly a whole day.
# When about finishing my internet broke down and this whole day scrap was in vain, so I decided to scrap 50000 tweets of every 3 months since 2021-04-21
scraped_tweets = sntwitter.TwitterSearchScraper('(COVID-19 vaccine OR Pfizer OR Moderna) min_replies:2 min_faves:5 lang:en until:2021-12-13 since:2021-10-20').get_items()

sliced_scraped_tweets = itertools.islice(scraped_tweets, 50000)
# Besides date and content, this code can scrap everything including the author and country, etc.
tweets_df = pd.DataFrame(sliced_scraped_tweets)
tweets_df2 = pd.DataFrame(tweets_df, columns=['date', 'content'])
tweets_df2['date'] = tweets_df2['date'].dt.tz_localize(None)
#print(tweets_df2)

tweets_df2.to_csv('D:/COVID-19-2110to2112.csv',  sep=',', index=False)