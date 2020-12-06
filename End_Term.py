# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 22:04:51 2020

@author: ajayr
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import textstat #from textstat import flesch_kincaid_grade
from textstat.textstat import textstat


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

import string

from collections import defaultdict
from collections import Counter

from wordcloud import WordCloud, STOPWORDS


# importing the required libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

color = sns.color_palette()


from nltk.stem import WordNetLemmatizer        
from nltk.classify import SklearnClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
import string
import nltk
nltk.download('wordnet')
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#df = pd.read_csv("amazon_reviews.txt", delimiter = "\t")
df= pd.read_csv('reviews.csv')
print(df.head())

# 0=Fake Reviews,1=Real Reviews
df.loc[df["LABEL"] == "__label1__", "LABEL"] = '0'
df.loc[df["LABEL"] == "__label2__", "LABEL"] = '1'


# Count Each Product by Cateogry
cnt_srs=df.groupby(df["LABEL"]).PRODUCT_CATEGORY.value_counts()
print(cnt_srs)

print(df.info())

cnt_rate = df.groupby(df["LABEL"]).RATING.value_counts()
print(cnt_srs)
sns.catplot('RATING',data=df,kind='count',hue='LABEL',order=[1,2,3,4,5],hue_order=['0','1'])

sns.catplot('RATING',data=df,kind='count',hue='PRODUCT_CATEGORY',order=[1,2,3,4,5])

sns.catplot('LABEL',data=df,kind='count',hue='PRODUCT_CATEGORY')


cnt_vp = df.groupby("VERIFIED_PURCHASE").LABEL.value_counts()
print(cnt_vp)


sns.catplot('LABEL',data=df,kind='count',hue='VERIFIED_PURCHASE')


df1 = df.groupby("LABEL").REVIEW_TEXT
df['TEXT_LENGTH'] = df['REVIEW_TEXT'].apply(len)
cnt_reviewtxt = df.groupby(["LABEL"]).TEXT_LENGTH.agg(lambda x: sum(x)/len(x))
print(cnt_reviewtxt)


plt.figure(figsize=(8,8))
sns.barplot(cnt_reviewtxt.index, cnt_reviewtxt.values, alpha=0.8,color='orange')
plt.ylabel('Text Length', fontsize=16)
plt.xlabel('Label', fontsize=16)
plt.title('Text length Vs Label', fontsize=18)
plt.xticks(rotation='horizontal')
plt.show()

df['num_sentences'] = df['REVIEW_TEXT'].apply(lambda x: len(str(x).split('.')))

df["FK_Score"] = df["REVIEW_TEXT"].apply(textstat.flesch_kincaid_grade)

print(df)

cnt_fkscore= df.groupby(["LABEL"]).FK_Score.agg(lambda x: sum(x)/len(x))
print(cnt_fkscore)

plt.figure(figsize=(8,8))
sns.barplot(cnt_fkscore.index, cnt_fkscore.values, alpha=0.8, color='blue')
plt.ylabel('FKscore', fontsize=16)
plt.xlabel('Label', fontsize=16)
plt.title('FKscore Vs Label', fontsize=18)
plt.xticks(rotation='horizontal')
plt.show()

# nltk.download('stopwords')
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def stopCount(x):
    sum =0
    for char in x.split():
        sum+= char in stop_words
    return sum
df['stop_count'] = df['REVIEW_TEXT'].apply(stopCount)


cnt_stp= df.groupby(["LABEL"]).stop_count.agg(lambda x: sum(x)/len(x))
print(cnt_stp)

plt.figure(figsize=(8,8))
sns.barplot(cnt_stp.index, cnt_stp.values, alpha=0.8, color=color[0])
plt.ylabel('Stopword counts', fontsize=16)
plt.xlabel('Label', fontsize=16)
plt.title('Stopwords Counts Vs Label', fontsize=18)
plt.xticks(rotation='horizontal')
plt.show()




def capsCount(x):
    sum =0
    for char in x:
        sum+= char in "QWERTYUIOPASDFGHJKLZXCVBNM"
    return sum
df['caps_count'] = df['REVIEW_TEXT'].apply(capsCount)
cnt_caps = df.groupby(["LABEL"]).caps_count.agg(lambda x: sum(x)/len(x))
print(cnt_caps)


#pie, ax = plt.subplots(figsize=[10,6])
#labels = cnt_caps.keys()
#plt.pie(x=cnt_caps, autopct="%.1f%%", labels=labels, pctdistance=0.5)
#plt.title("Caps_count Vs Label", fontsize=14);

plt.figure(figsize=(8,8))
sns.barplot(cnt_caps.index, cnt_caps.values, alpha=0.8, color=color[0])
plt.ylabel('Caps_count', fontsize=16)
plt.xlabel('Label', fontsize=16)
plt.title('Caps_count Vs Label', fontsize=18)
plt.xticks(rotation='horizontal')
plt.show()

count = lambda l1,l2: sum([1 for x in l1 if x in l2])
def punctCount(x):
    return count(x, set(string.punctuation))
df['punct_count'] = df['REVIEW_TEXT'].apply(punctCount)
cnt_punc = df.groupby(["LABEL"]).punct_count.agg(lambda x: sum(x)/len(x))
print(cnt_punc)


plt.figure(figsize=(8,8))
sns.barplot(cnt_punc.index, cnt_punc.values, alpha=0.8, color=color[0])
plt.ylabel('Punctuation_count', fontsize=16)
plt.xlabel('Label', fontsize=16)
plt.title('Punctuation_count Vs Label', fontsize=18)
plt.xticks(rotation='horizontal')
plt.show()

match_list = []

def checkName(title,text):
    matches = []
    for word in title.split():
        #removing punctuation
        word = "".join((char for char in word if char not in string.punctuation))
        #print(word)
        myreg = r'\b'+word+r'\b'
        r = re.compile(myreg, flags=re.I | re.X)
        matches.append(r.findall(text))
    return len(matches)
        

for a,b in zip(df.PRODUCT_TITLE, df.REVIEW_TEXT):
    number_of_matches = checkName(a,b)
    match_list.append(number_of_matches)
    
df["matchesDf"] = match_list


cnt_matchdf = df.groupby(["LABEL"]).matchesDf.agg(lambda x: sum(x)/len(x))
print(cnt_matchdf)


plt.figure(figsize=(8,8))
sns.barplot(cnt_matchdf.index, cnt_matchdf.values, alpha=0.8, color=color[3])
plt.ylabel('ProductName_Matches_count', fontsize=16)
plt.xlabel('Label', fontsize=16)
plt.title('ProductName_Matches_count Vs Label', fontsize=18)
plt.xticks(rotation='horizontal')
plt.show()


fake_rev=[]
real_rev=[]

for i in df.DOC_ID:
    if i<=10500:
        fake_rev.append(df.REVIEW_TEXT[i-1])
    else:
        real_rev.append(df.REVIEW_TEXT[i-1])


fake_rev_cln =re.sub("[^a-zA-Z]"," ",str(fake_rev))
real_rev_cln =re.sub("[^a-zA-Z]"," ",str(real_rev))
#print(real_rev_cln)
# creating and presenting the wordcloud
stpwords=['one','fit','even','well','br','say','now','thing','make','set','watch','will','size','got','don','take']+list(STOPWORDS)
def create_word_cloud(txt):
    cloud = WordCloud(background_color='black',stopwords=stpwords,
                       colormap="Blues", min_font_size=12).generate(txt)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    
create_word_cloud(fake_rev_cln)
create_word_cloud(real_rev_cln)


fake_rev=[]
real_rev=[]

for i in df.DOC_ID:
    if i<=10500:
        fake_rev.append(df.REVIEW_TEXT[i-1])
    else:
        real_rev.append(df.REVIEW_TEXT[i-1])

analyzer = SentimentIntensityAnalyzer()

sentiment = df['REVIEW_TEXT'].apply(analyzer.polarity_scores)
sentiment_df = pd.DataFrame(sentiment.tolist())

print(sentiment_df)


df_sentiment = pd.concat([df,sentiment_df], axis = 1)
print(df_sentiment.head())

fake_neg_score=[]
real_neg_score=[]
for i in df.DOC_ID:
    if i<=10500:
        fake_neg_score.append(df_sentiment.neg[i-1])
    else:
        real_neg_score.append(df_sentiment.neg[i-1])

fake_pos_score=[]
real_pos_score=[]
for i in df.DOC_ID:
    if i<=10500:
        fake_pos_score.append(df_sentiment.pos[i-1])
    else:
        real_pos_score.append(df_sentiment.pos[i-1])
        
        
        
fake_neu_score=[]
real_neu_score=[]
for i in df.DOC_ID:
    if i<=10500:
        fake_neu_score.append(df_sentiment.neg[i-1])
    else:
        real_neu_score.append(df_sentiment.neu[i-1])
        
        
        
fake_comp_score=[]
real_comp_score=[]
for i in df.DOC_ID:
    if i<=10500:
        fake_comp_score.append(df_sentiment.compound[i-1])
    else:
        real_comp_score.append(df_sentiment.compound[i-1])

print('Fake Review Overall Positive Sentiment:',np.mean(fake_pos_score))
print('Real Review Overall Positive Sentiment:',np.mean(real_pos_score))

print('Fake Review Overall Negative Sentiment:',np.mean(fake_neg_score))
print('Real Review Overall Negative Sentiment:',np.mean(real_neg_score))


print('Fake Review Overall Neutral Sentiment:',np.mean(fake_neu_score))
print('Real Review Overall Neutral Sentiment:',np.mean(real_neu_score))

print('Fake Review Overall Compound Sentiment:',np.mean(fake_comp_score))
print('Real Review Overall Compound Sentiment:',np.mean(real_comp_score))




#getting the basic information about the file
df.info()


#importing punctuations to be removed from reviews
punct = string.punctuation
print(punct)


#converting reviews to list of reviews
list_of_reviews = df.REVIEW_TEXT.to_list()

#converting reviews to lower case
list_of_reviews = [doc.lower() for doc in list_of_reviews]


# words tokenization
from nltk.tokenize import word_tokenize
words_tok = [word_tokenize(doc) for doc in list_of_reviews]
print(words_tok)


#sentence tokeization
from nltk.tokenize import sent_tokenize
sent_tok = [sent_tokenize(doc) for doc in list_of_reviews]
print(sent_tok)



"# Removing punctuation\n"
import re
regex = re.compile('[%s]' % re.escape(string.punctuation))
tokenized_docs_no_punctuation = []

for review in words_tok:
    new_review = []
    for token in review:
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            new_review.append(new_token)
        
    tokenized_docs_no_punctuation.append(new_review)
print(tokenized_docs_no_punctuation)




#removing stopwords
from nltk.corpus import stopwords
tokenized_docs_no_stopwords = []

for doc in tokenized_docs_no_punctuation:
    new_term_vector = []
    for word in doc:
        if not word in stopwords.words('english'):
            new_term_vector.append(word)
    tokenized_docs_no_stopwords.append(new_term_vector)
print(tokenized_docs_no_stopwords)



#creating a single list
import itertools
chain_object = itertools.chain.from_iterable(tokenized_docs_no_stopwords)
flattened_list = list(chain_object)
print(flattened_list)


# using list comprehension to convert to string
listToStr = ' '.join([str(elem) for elem in flattened_list]) 
  
print(listToStr)  


#importing libraries for tfidf 
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
tfidf = tfidf.fit(flattened_list)

#focus on idf values
print(tfidf.idf_)

#summarize
print(tfidf.vocabulary_)

input = ['when least you think so', 'only giving this 3 stars because it is so cheap.']

vector = tfidf.transform(input)



print(vector.toarray())

data = pd.read_csv('reviews.csv')
data["LABEL"]= data["LABEL"].replace("__label1__", 0) 
data["LABEL"]= data["LABEL"].replace("__label2__", 1) 

data = data[['REVIEW_TEXT', 'LABEL']]
x = data['REVIEW_TEXT']
y = data['LABEL']


punct = string.punctuation


table = str.maketrans({key: None for key in string.punctuation})
def text_data_cleaning(text):
    # Should return a list of tokens
    lemmatizer = WordNetLemmatizer()
    filtered_tokens=[]
    lemmatized_tokens = []
    stop_words = set(stopwords.words('english'))
    text = text.translate(table)
    for w in text.split(" "):
        if w not in stop_words:
            lemmatized_tokens.append(lemmatizer.lemmatize(w.lower()))
        filtered_tokens = [' '.join(l) for l in nltk.bigrams(lemmatized_tokens)] + lemmatized_tokens
    return filtered_tokens

text_data_cleaning("Hello all, It's a beautiful day outside there!")

tfidf = TfidfVectorizer(tokenizer=text_data_cleaning, lowercase=False, ngram_range=(1,1))

classifier = LinearSVC(C=1.0)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


x_train.shape, x_test.shape


clf = Pipeline([('tfidf',tfidf), ('clf',classifier)])

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(y_pred)

confusion_matrix(y_test, y_pred)
print(classification_report(y_test,y_pred))

accuracy_score(y_test, y_pred)

clf.predict(["Purchased this after reading multiple good reviews. Unfortunately for me this didn't work well at all. It's a couple inches too short so my neck kinks and muscles pull and spasm as a result. I figured it was an issue with my neck so I brought it to my chiropractor appointment to find out if it's me or the pillow. He suggested I measure from my neck to my shoulder and find a pillow that the height is the same. Even when I lie ony back, the pillow is way off and creates additional discomfort. Such as burning muscles around my head. Jaw problems, pain and tightness. And I always end up with a kink in my neck. I really wanted to love this pillow. Then I would've settled on simply being able to sleep a little better but it's made everything worse. To attempt to compensate for the lack of height I've lined my bed with a folded towel. Didn't help. Finally I found a really thin soft pillow, which helped a little. I hate how much money I loose to pillows that are either completely terrible or not for my needs, that simply don't make me any more comfortable but more uncomfortable. I think this would be great for a youth. Because I'm a 5' tall adult and it's far too small"])

precision_recall_fscore_support(y_test, y_pred, average='weighted')

data = pd.read_csv('reviews.csv')
data["LABEL"]= data["LABEL"].replace("__label1__", 0) 
data["LABEL"]= data["LABEL"].replace("__label2__", 1) 

data = data[['REVIEW_TEXT', 'LABEL']]
x = data['REVIEW_TEXT']
y = data['LABEL']


punct = string.punctuation
punct


table = str.maketrans({key: None for key in string.punctuation})
def text_data_cleaning(text):
    # Should return a list of tokens
    lemmatizer = WordNetLemmatizer()
    filtered_tokens=[]
    lemmatized_tokens = []
    stop_words = set(stopwords.words('english'))
    text = text.translate(table)
    for w in text.split(" "):
        if w not in stop_words:
            lemmatized_tokens.append(lemmatizer.lemmatize(w.lower()))
        filtered_tokens = [' '.join(l) for l in nltk.bigrams(lemmatized_tokens)] + lemmatized_tokens
    return filtered_tokens

text_data_cleaning("Hello all, It's a beautiful day outside there!")

tfidf = TfidfVectorizer(tokenizer=text_data_cleaning, lowercase=False, ngram_range=(1,1))

classifier = MultinomialNB(alpha=1.5, class_prior=None, fit_prior=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


x_train.shape, x_test.shape


clf = Pipeline([('tfidf',tfidf), ('clf',classifier)])

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(y_pred)

confusion_matrix(y_test, y_pred)
print(classification_report(y_test,y_pred))

accuracy_score(y_test, y_pred)

clf.predict(["Purchased this after reading multiple good reviews. Unfortunately for me this didn't work well at all. It's a couple inches too short so my neck kinks and muscles pull and spasm as a result. I figured it was an issue with my neck so I brought it to my chiropractor appointment to find out if it's me or the pillow. He suggested I measure from my neck to my shoulder and find a pillow that the height is the same. Even when I lie ony back, the pillow is way off and creates additional discomfort. Such as burning muscles around my head. Jaw problems, pain and tightness. And I always end up with a kink in my neck. I really wanted to love this pillow. Then I would've settled on simply being able to sleep a little better but it's made everything worse. To attempt to compensate for the lack of height I've lined my bed with a folded towel. Didn't help. Finally I found a really thin soft pillow, which helped a little. I hate how much money I loose to pillows that are either completely terrible or not for my needs, that simply don't make me any more comfortable but more uncomfortable. I think this would be great for a youth. Because I'm a 5' tall adult and it's far too small"])

precision_recall_fscore_support(y_test, y_pred, average='weighted')