import numpy as np
import pandas as pd
import io
import requests
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from textblob import TextBlob, Word
import sklearn.ensemble as skens

# ----------------------------------------------------------------------------
# 1. load datasets
# ----------------------------------------------------------------------------

# get adjusted IBM minitely stock price on April, 9, 2009
ibm_url = 'https://raw.githubusercontent.com/xinyexu/SI618-Individual-Project/master/IBM_Adjusted_20090407.csv'
ibm_string_file = requests.get(ibm_url).content
ibm = pd.read_csv(io.StringIO(ibm_string_file.decode('utf-8')))

# get twitter data on April 7, 8, 19, on April, 2009
twitter_pos = '/Users/xuxinye/Desktop/SI 618 Project/trainingandtestdata/twitter information.csv'
twitter = pd.read_csv(twitter_pos, header=None, engine='python')
twitter.columns = ['Polarity', 'ID', 'Date', 'Query', 'User', 'Tweet']
# engine = 'python' to avoid errors; or encoding='latin1',
# https://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas-with-python

# trasfer string date into date format (slowly)
twitter.loc[:,'Date'] = pd.to_datetime(twitter.Date, errors = 'ignore')
print(twitter.Date.describe())

# ----------------------------------------------------------------------------
# 2. explore dataframe
# ----------------------------------------------------------------------------

# show the length of tweet
sns.set_style("whitegrid", {'axes.grid' : True})
sns.distplot(twitter.Tweet.str.len(), kde=False).set_title('Tweet Length Dist')
plt.xlim(0, 175)
plt.show(block=True)

# groupby different day to check dates
check_date = twitter.set_index('Date').groupby(pd.Grouper(freq='D')).count()
check_date_wona = check_date[(check_date.T != 0).any()]
# show the Number of Tweet
plt.bar(check_date_wona.index, check_date_wona.Tweet)
plt.xlabel('Date')
plt.ylabel('Number of Tweet')
plt.title('Number of Tweet of each date')
plt.show(block=True)

# show the distribution of sentiment
sns.distplot(twitter.Polarity).set_title('Polarity dist during this period')
plt.show(block=True)

# ----------------------------------------------------------------------------
# 3. find the best way to estimate sentiment
# ----------------------------------------------------------------------------

# sort weitter dataset by time, ascending
twitter_asc = twitter.sort_values(by=['Date'], ascending=True)
twitter_asc = twitter_asc.reset_index(drop=True) # drop original indexs

# seperate train (80% dates) and test by its indexs:
train_end = check_date_wona.index[round(len(check_date_wona)*0.8)] # Timestamp('2009-06-16 00:00:00')
train_end_index = twitter_asc[twitter_asc.loc[:, 'Date'] > train_end].index[0] #  758080
twitter_train = twitter_asc.loc[0:train_end_index-1,:]
twitter_test = twitter_asc.loc[train_end_index:,:]
print('length of twitter_train', len(twitter_train), ';length of twitter_test', len(twitter_test),
      ';ratio of total train data/test', len(twitter_train)/len(twitter_asc))


# (a) Naive Bayes
# create text vectorizer
# vectorizer = CountVectorizer(min_df=.001, max_df=.8, stop_words='english')
vectorizer = CountVectorizer(stop_words='english')
train_dtm = vectorizer.fit_transform(twitter_train.Tweet)
test_dtm = vectorizer.transform(twitter_test.Tweet)

# train Naive Bayes
nb = MultinomialNB()
nb.fit(train_dtm, twitter_train.Polarity)

# Evaluate Results
pred_polarity = nb.predict(test_dtm)
accuracy_score(twitter_test.Polarity, pred_polarity) # 0.9242709021810972
# class_log_prior_
print(nb.class_log_prior_)


# (b) TextBlob package for sentiment
# example:
sample = twitter_train.Tweet[0]
print('Tweet: ',sample)
sample_est = TextBlob(sample).polarity
print('Real Polarity: ', twitter_train.Polarity[0],  '; Est Polarity: ', sample_est)

# otehr example
# train[['text']].sample(10).assign(sentiment=lambda x: x.text.apply(estimate_polarity)).sort_values('sentiment')

# Evaluate Results for TextBlob
# rescale textbolb plarity the [-1, 1] to 0 or 4
def textbolb_scale(text):
    tetbolb_pol = TextBlob(text).polarity
    if tetbolb_pol > 0:
        return(4)
    else:
        return(0)
    # ignore neutral here

pred_polarity_textblob = twitter_test.Tweet.apply(textbolb_scale)
accuracy_score(twitter_test.Polarity, pred_polarity_textblob) # 0.688956130204891

# another way even without scale:
# test_set = twitter_test.Tweet.apply(lambda x: TextBlob(x).sentiment.polarity)
# accuracy_score(twitter_test.Polarity, (test_set>0).astype(int))

# (c) Random Forest
rf_model = skens.RandomForestClassifier(n_estimators=10, oob_score=False, criterion='entropy')
# oob_score: whether to use out-of-bag samples to estimate the R^2 on unseen data.
rf_model.fit(train_dtm, twitter_train.Polarity)

# Evaluate Results for
pred_polarity_RF = rf_model.predict(test_dtm)
accuracy_score(twitter_test.Polarity, pred_polarity_RF) # 0.9242709021810972


#
# # find the best NB model with optimal parameters
# params = {
#                  'n_estimators': [5, 10, 15, 20, 25],
#                  'max_depth': [2, 5, 7, 9],
#              }
# NB_CV = GridSearchCV(nb, cv=10, param_grid=params) # , iid = True
# NB_CV.fit(train_dtm,twitter_train.Polarity)
# print(NB_CV.cv_results_['mean_test_score'])









#############################
# important reference:
# https://github.com/PacktPublishing/Hands-On-Machine-Learning-for-Algorithmic-Trading/blob/master/Chapter13/04_text_classification.ipynb




#############################
# # other way to get data
# import requests
# import pandas as pd
# import arrow
# from dateutil.parser import parse
# from dateutil.tz import gettz
# import datetime
# from pprint import pprint
# import urllib,time,datetime
# import sys
#
# symbol1 = sys.argv[1]
# symbolname = symbol1
# symbol1 = symbol1.upper()
#
# def get_quote_data(symbol='iwm', data_range='1d', data_interval='1m'):
#     res = requests.get(
#         'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={data_range}&interval={data_interval}'.format(
#             **locals()))
#     data = res.json()
#     body = data['chart']['result'][0]
#     dt = datetime.datetime
#     dt = pd.Series(map(lambda x: arrow.get(x).to('EST').datetime.replace(tzinfo=None), body['timestamp']), name='dt')
#     df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
#     dg = pd.DataFrame(body['timestamp'])
#     return df.loc[:, ('open', 'high', 'low', 'close', 'volume')]
#
# q = jpy5m = get_quote_data('^GSPC', '2d', ' 1m')
# print q
#
#
# # another way:
#
# import pandas_datareader as pdr
# from datetime import datetime
#
# appl = pdr.get_data_yahoo(symbols='^GSPC', start=datetime(2000, 1, 1), end=datetime(2012, 1, 1))
# print(appl['Adj Close'])






