#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:25:11 2020

@author: gracemcmonagle
"""
#%%
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

filepath = '/Users/gracemcmonagle/Desktop/School/Fall 2020/EECS 731/Project 2/1028_2124_bundle_archive/Shakespeare_data.csv'
rawData = pd.read_csv(filepath, delimiter = ',')

#%%
#remove entries that aren't lines (acts, scenes)

data = rawData.dropna()
players = data.Player.unique().tolist()

#create dummy variables for plays
playsdf = pd.get_dummies(data, columns=['Play'])
data = playsdf


#split ActSceneLine into three columns
for index, row in data.iterrows():
    data['ActSceneLine'][index] = data['ActSceneLine'][index].split('.')
        
act = []
scene = []
line = []
for index, row  in data.iterrows():
    act.append(data['ActSceneLine'][index][0])
    scene.append(data['ActSceneLine'][index][1])
    line.append(data['ActSceneLine'][index][2])
    
data['Act'] = act
data['Scene'] = scene
data['Line'] = line


#count the number of words in each line
noWords = []
for index, row in data.iterrows():
    noWords.append(len(row['PlayerLine'].split()))
    
data['NumWords'] = noWords

#%% Count all the words, and sort by most used words

wordCount = {}
for index, row in data.iterrows():
    line = row['PlayerLine'].translate(str.maketrans('', '', string.punctuation)).lower()
    words = line.split()
    for word in words:
        if word not in wordCount:
            wordCount[word] = 0
        wordCount[word] += 1
        
sorted_wordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)
#top 50 words
top_words50 = sorted_wordCount[:50]
top_words100 = sorted_wordCount[:100]


#%% Bar graph of top 50 words
plt.bar(range(len(top_words50)), [val[1] for val in top_words50], align='center')
plt.xticks(range(len(top_words50)), [val[0] for val in top_words50])
plt.xticks(rotation=70)
plt.draw()
#%% Make new columns for each of the top 50 words

columns = {}
for word in top_words100:
    columns[word[0]] = [0 for _ in range(len(noWords))]

i=0   
for index, row in data.iterrows():
    line = row['PlayerLine'].translate(str.maketrans('', '', string.punctuation)).lower()
    words = line.split()
    for word in words:
        if word in columns:
            columns[word][i] += 1
    i+=1
    
for word in top_words100:
    data[word[0] + '_count'] = columns[word[0]]


#%% testing 
labels = data['Player']   
data_x = data.drop(['Player', 'Dataline','ActSceneLine', 'PlayerLine'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(data_x.head(10000), labels.head(10000), test_size=0.2, random_state=0)



    
#%% Create train and test datasets
    
labels = data['Player']
data_playerLine = data['PlayerLine']
data_x = data.drop(['Player', 'Dataline', 'Play','ActSceneLine', 'PlayerLine'], axis=1)

#cv = CountVectorizer(binary = False, max_df = .95)
#cv.fit_transform(data_playerLine)
#data_playerLineTransform = cv.transform(data_playerLine.values)
#playerArray = data_playerLineTransform.toarray()
#playerDF = pd.DataFrame(np.row_stack(playerArray))

#fullData_x = pd.concat([data_x, playerDF.set_index(data_x.index)], axis=1)

X_train, X_test, y_train, y_test = train_test_split(data_x, labels, test_size=0.2, random_state=0)

#tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
#tfidf.fit(X_train['PlayerLine'].to_numpy())
#X_train_vect= X_train
#X_test_vect = X_test
#X_train_vect['PlayerLine'] = tfidf.transform(X_train['PlayerLine'].to_numpy())
#X_test_vect['PlayerLine'] = tfidf.transform(X_test['PlayerLine'].to_numpy())

    
#%%

classifier = LogisticRegression()
classifier.fit(X_train,y_train)
score = classifier.score(X_test, y_test)
print(score)
#%%


    