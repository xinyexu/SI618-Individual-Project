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
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import datetime
from scipy import stats
import pandas_datareader as pdr # conenct yahoo finance
from datetime import datetime
import gensim # word2vec model for text similarity

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
# find weekdays: Monday is 0 and Sunday is 6
check_date_wona.loc[:, 'Weekdays'] = [check_date_wona.index[i].weekday() for i in range(len(check_date_wona))]
print('Check weekdays', '\n', check_date_wona.Weekdays.value_counts())


# show the Number of Tweet
plt.bar(check_date_wona.index, check_date_wona.Tweet)
plt.xlabel('Date')
plt.ylabel('Number of Tweet')
plt.title('Number of Tweet of each date')
plt.show(block=True)

# show common words in people's tweet by word cloud
text = ''
for sens in twitter.Tweet:
    text += sens
wordcloud = WordCloud(max_font_size=40).generate(text)
plt.figure(figsize=(25,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show(block=True)

# show the distribution of sentiment in each date
from collections import Counter
twitter_cate = twitter.copy()
twitter_cate.loc[:,'Polarity'] = twitter_cate.loc[:,'Polarity'].astype(object)
twitter_cate = pd.get_dummies(twitter_cate, columns=['Polarity'])
check_senti = twitter_cate.set_index('Date').groupby(pd.Grouper(freq='D'))['Polarity_0', 'Polarity_4'].sum()
check_summary = pd.concat([check_date_wona, check_senti], axis = 1, join = 'inner')
check_summary.loc[:, 'Pol_0_rat'] = check_summary.Polarity_0 / check_summary.Polarity
check_summary.loc[:, 'Pol_4_rat'] = check_summary.Polarity_4 / check_summary.Polarity

x = check_summary.index
ax = plt.subplot(111)
ax.set_title('Polarity Neg vs Pos', fontsize=10)
ax.bar(x, check_summary.Pol_0_rat,color='b',align='center', label='Polarity Neg')
ax.bar(x, check_summary.Pol_4_rat, bottom=check_summary.Pol_0_rat, color='y',align='center', label='Polarity Pos')
ax.xaxis_date()
plt.legend(bbox_to_anchor=(0.05,0.8), loc='center left')
# plt.plot(check_summary.index, check_summary[['Polarity_0', 'Polarity_4']], kind="bar")
# sns.barplot(check_summary.index, check_summary[['Polarity_0', 'Polarity_4']]).set_title('Polarity dist during this period')
plt.show(block=True)

# We found the data after 2009-05-29 only contains negative sentiment
check_short = check_summary[check_summary.index<='2009-05-28']
print(len(check_short))
print(check_short.Pol_4_rat.mean())
print('check weekdays', '\n', check_short.Weekdays.value_counts())


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

# create text vectorizer
# vectorizer = CountVectorizer(min_df=.001, max_df=.8, stop_words='english')
vectorizer = CountVectorizer(stop_words='english')
train_dtm = vectorizer.fit_transform(twitter_train.Tweet)
test_dtm = vectorizer.transform(twitter_test.Tweet)

# (a) Naive Bayes
# train Naive Bayes
nb = MultinomialNB()
nb.fit(train_dtm, twitter_train.Polarity)

# Evaluate Results
# train accuracy
pred_polarity_train = nb.predict(train_dtm)
accuracy_score(twitter_train.Polarity, pred_polarity_train) # 0.8509537252005065

# test accuracy
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

# other example
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
# train
pred_polarity_textblob_train = twitter_train.Tweet.apply(textbolb_scale)
accuracy_score(twitter_train.Polarity, pred_polarity_textblob_train) # 0.6460162515829464

# test
pred_polarity_textblob = twitter_test.Tweet.apply(textbolb_scale)
accuracy_score(twitter_test.Polarity, pred_polarity_textblob) # 0.688956130204891

# another way even without scale:
# test_set = twitter_test.Tweet.apply(lambda x: TextBlob(x).sentiment.polarity)
# accuracy_score(twitter_test.Polarity, (test_set>0).astype(int))

# (c) Random Forest

# test coding:
# train_dtm = vectorizer.fit_transform(twitter_train.Tweet[:100])
rf_model = skens.RandomForestClassifier(n_estimators=10, oob_score=False, criterion='entropy')
# oob_score: whether to use out-of-bag samples to estimate the R^2 on unseen data.
rf_model.fit(train_dtm, twitter_train.Polarity)

# Evaluate Results for
# train accuracy
pred_polarity_train_RF = rf_model.predict(train_dtm)
accuracy_score(twitter_train.Polarity, pred_polarity_train_RF) # 0.8242256753904601
# test accuracy
pred_polarity_RF = rf_model.predict(test_dtm)
accuracy_score(twitter_test.Polarity, pred_polarity_RF) # 0.8706316093853271

# find the best RF model with optimal parameters
params = {
                 'n_estimators': [5, 10, 15, 20, 25],
                 # 'max_depth': [2, 5, 7, 9],
             }
RF_CV = GridSearchCV(rf_model, cv=10, param_grid=params) # , iid = True
RF_CV.fit(train_dtm,twitter_train.Polarity)
print(RF_CV.best_estimator_)
print(RF_CV.cv_results_['mean_test_score'])
print(RF_CV.cv_results_['mean_test_score'].max())



# # find the best NB model with optimal parameters
# params = {
#                  'n_estimators': [5, 10, 15, 20, 25],
#                  'max_depth': [2, 5, 7, 9],
#              }
# NB_CV = GridSearchCV(nb, cv=10, param_grid=params) # , iid = True
# NB_CV.fit(train_dtm,twitter_train.Polarity)
# print(NB_CV.cv_results_['mean_test_score'])





# ----------------------------------------------------------------------------
# 4. relationship between IBM stock price and all tweet sentiment
# ----------------------------------------------------------------------------


# for train dates, using true polarity
# add Date and Time together
ibm.loc[:, 'DateTime'] = pd.to_datetime(ibm.Date + ' ' + ibm.Time)
ibm = ibm.set_index('DateTime')

# average the tweet polarity in minutes, mean is 2 (4 for pos, 0 for neg)!
# pd.groupby frequencies: http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
twitter_1min = twitter.set_index('Date').groupby(pd.Grouper(freq='min')).Polarity.mean()
# We found the data after 2009-05-29 only contains negative sentiment, drop them
twitter_1min_short = twitter_1min[twitter_1min.index<='2009-05-28']
twitter_1min_wona = twitter_1min_short.dropna()

# merge with twitter dataset sentiment average for the previous minnus
ibm_and_twe = pd.merge(pd.DataFrame(ibm.Close), pd.DataFrame(twitter_1min_wona),
                       left_index=True, right_index=True, how='outer')
ibm_and_twe_wona = ibm_and_twe.dropna()
ibm_and_twe = pd.concat([ibm.Close, twitter_1min_wona], axis = 1, join = 'inner')

# we find the most tweets in above datasets happened durinig the night, without trading!!!


# summary them into daily sentiment score
twitter_1d = twitter.set_index('Date').groupby(pd.Grouper(freq='D')).Polarity.mean()
# We found the data after 2009-05-29 only contains negative sentiment, drop them
twitter_1d_short = twitter_1d[twitter_1d.index<='2009-05-28']
twitter_1d_wona = twitter_1d_short.dropna()
# merge with ibm price,

# get ibm daily price
ibm_daily = pdr.get_data_yahoo(symbols='IBM', start=datetime(2009, 4, 6), end=datetime(2009, 5, 28))

# merge with twitter dataset sentiment average for the previous day
ibm_dialy_twe = pd.concat([ibm_daily[['Adj Close']], twitter_1d_wona], axis = 1, join = 'inner')

# find the optimal lag days of Polarity with Adj Close by spearman correlation
# update lag days Polarity by moving average with previous lag days
spearman = []
for i in range(10):
    pol_lag = ibm_dialy_twe[['Polarity']].rolling(window=i+1).mean()
    spearman.append(stats.spearmanr(pol_lag, ibm_dialy_twe[['Adj Close']], nan_policy='omit')[0])
optimal_roll = spearman.index(max(spearman)) + 1
print('max spearman: ', max(spearman), 'optimal lag: ', optimal_roll)


# correlation pearson and spearman
# lag Polarity one day
ibm_dialy_twe.loc[:,'Polarity'] = ibm_dialy_twe[['Polarity']].shift(1)
g = sns.JointGrid(data=ibm_dialy_twe,x='Adj Close',y='Polarity')
g = g.plot(sns.regplot, sns.distplot)
# g = g.set(xlabel='Adj Close', ylabel='Lag 1 Polarity')
g = g.annotate(stats.spearmanr)
plt.ylabel("Lag 1 Polarity")
plt.show(block=True)

# lag Polarity five day
pol_lag = ibm_dialy_twe[['Polarity']].rolling(window=optimal_roll).mean()
g = sns.JointGrid(ibm_dialy_twe[['Adj Close']], pol_lag)
g = g.plot(sns.regplot, sns.distplot)
g = g.annotate(stats.spearmanr)
plt.xlabel("Adj Close")
plt.ylabel("Lag 8 Polarity")
plt.show(block=True)


# ----------------------------------------------------------------------------
# 5. relationship between IBM stock price and relevant tweet sentiment
# ----------------------------------------------------------------------------

# Find More Relevant Tweets with IBM
# set IBM key words to filter relevant tweets (add 'IBM' itself)
# IBM industry:  https://www.ibm.com/industries?lnk=min
ibm_key = 'Aerospace, defense, Automotive, Banking, financial markets, Chemicals, Construction, Education, ' \
          'Electronics, Energy, utilities, Government, Healthcare, Insurance, Life, sciences, Manufacturing,' \
          ' Metals, mining, Oil, gas, Retail, Consumer, Products, Telecommunications, media, entertainment, ' \
          'Travel, transportation, IBM'

# short the tweet dataset to date
twitter_bef_0528 = twitter_asc[twitter_asc.Date<='2009-05-28']
print(len(twitter_bef_0528))

# pre-train mode
# load pre-trained word2vec model
# from https://github.com/eyaler/word2vec-slim
w2v_mod = gensim.models.KeyedVectors.load_word2vec_format("/Users/xuxinye/Downloads/618_07_NLP/"
                                                          "GoogleNews-vectors-negative300-SLIM.bin", binary=True)

# Train my own word2vec vector.
# a sequence of sentences as its input. Each sentence a list of words (utf8 strings):
sentences = [sens.split() for sens in twitter_bef_0528.Tweet]
sentences.append(ibm_key.split()) # add ibm words
mymodel = gensim.models.Word2Vec(sentences, min_count=10)

# by pre train model
total_score = []
for row in range(3):
    score = []
    for i in twitter_bef_0528.Tweet[row].split():
        for ibm_word in ibm_key:
            try:
                score.append(w2v_mod.similarity(i, ibm_word))
            except KeyError:
                score.append(0)
    total_score.append(np.mean(score))
print(total_score)
# [0.0370971685863098, 0.030408827317550534, 0.03274327167493441]

# words not in vocabulary
# words not in a little more sophisticated approach has been implemented in fastText
# (now also integrated into gensim): break the unknown word into smaller character n-grams.
# Assemble the word vector from vectors of these ngrams.
# https://radimrehurek.com/gensim/models/fasttext.html

total_score = []
for row in range(3):
    score = []
    for i in twitter_bef_0528.Tweet[row].split():
        for ibm_word in ibm_key:
            try:
                score.append(mymodel.similarity(i, ibm_word))
            except KeyError:
                score.append(0)
    total_score.append(np.mean(score))
print(total_score)
# [0.050589945092463284, 0.05649464393900006, 0.036852518259192654]

# use my model for total tweests, too slow, need improvement
total_score = []
for row in range(len(twitter_bef_0528)):
    score = []
    for i in twitter_bef_0528.Tweet[row].split():
        for ibm_word in ibm_key:
            try:
                score.append(mymodel.similarity(i, ibm_word))
            except KeyError:
                score.append(0)
    total_score.append(np.mean(score))

twitter_simi_ibm = pd.concat([twitter_bef_0528,  pd.Series(total_score)], axis = 1, join = 'inner')
# twitter_simi_ibm.to_csv('twitter_simi_score.csv', index=False)
twitter_simi_ibm = twitter_simi_ibm.rename(columns={0: 'Similarity_ibm'})

# find statistics of similarity distribution
print(twitter_simi_ibm.Similarity_ibm.describe())
benchmarkt = np.mean(twitter_simi_ibm.Similarity_ibm)
twitter_simi_ibm_overmean = twitter_simi_ibm[twitter_simi_ibm.Similarity_ibm >= benchmarkt]

# summary them into daily sentiment score
twitter_1d_ibm = twitter_simi_ibm_overmean.set_index('Date').groupby(pd.Grouper(freq='D')).Polarity.mean()
# We found the data after 2009-05-29 only contains negative sentiment, drop them
twitter_1d_ibm_wona = twitter_1d_ibm.dropna()

# merge with twitter dataset sentiment average for the previous day
ibm_releated_twe = pd.concat([ibm_daily[['Adj Close']], twitter_1d_ibm_wona], axis = 1, join = 'inner')

# find the optimal lag days of Polarity with Adj Close by spearman correlation
# update lag days Polarity by moving average with previous lag days
spearman = []
for i in range(12):
    pol_lag = ibm_releated_twe[['Polarity']].rolling(window=i+1).mean()
    spearman.append(stats.spearmanr(pol_lag, ibm_releated_twe[['Adj Close']], nan_policy='omit')[0])
optimal_roll = spearman.index(max(spearman)) + 1
print('max spearman: ', max(spearman), 'optimal lag: ', optimal_roll)


# correlation pearson and spearman
# lag Polarity one day
ibm_releated_twe.loc[:,'Polarity'] = ibm_releated_twe[['Polarity']].shift(1)
g = sns.JointGrid(data=ibm_releated_twe,x='Adj Close',y='Polarity')
g = g.plot(sns.regplot, sns.distplot)
# g = g.set(xlabel='Adj Close', ylabel='Lag 1 Polarity')
g = g.annotate(stats.spearmanr)
plt.ylabel("Lag 1 Polarity")
plt.show(block=True)

# lag Polarity optimal lag day
pol_lag = ibm_releated_twe[['Polarity']].rolling(window=optimal_roll).mean()
g = sns.JointGrid(ibm_releated_twe[['Adj Close']], pol_lag)
g = g.plot(sns.regplot, sns.distplot)
g = g.annotate(stats.spearmanr)
plt.xlabel("Adj Close")
plt.ylabel("Lag 10 Polarity")
plt.show(block=True)





# alternative data sets:
# load new dataset with sentiment score and ibm price
ibm_url_new = 'https://raw.githubusercontent.com/aficionado/upordown/master/data/ibm_sentiment.csv'
ibm_string_file_new = requests.get(ibm_url_new).content
ibm_new = pd.read_csv(io.StringIO(ibm_string_file_new.decode('utf-8')))

sns.pairplot(ibm_new[['Volume', 'Close', 'Bullish', 'Bearish']])
plt.show(block=True)



