# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 12:06:18 2021

@author: ratho
"""

import pandas as pd
df = pd.read_csv('newsData.csv', encoding='ISO-8859-1')

#splitting data set into train and test
train = df[df['Date']<'20150101']
test = df[df['Date']>'20141231']

#data cleaning
#removing punctuation and other things
data = train.iloc[:,2:27]
data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

#converting all headlines to lowercase
#length = len(data.index)
for i in range(0, len(data.index)):
    data.iloc[i,:] = data.iloc[i,:].str.lower()

#considering all 25 headlines as a single paragraph
headlines = []
for i in range(0, len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[i,:]))

#creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(2,2))
train_dataset = cv.fit_transform(headlines)

#training the model using RandomForest
from sklearn.ensemble import RandomForestClassifier
#implementing RandomForest classifier
rfc_model = RandomForestClassifier(n_estimators=200, criterion='entropy')
rfc_model.fit(train_dataset,train['Label'])

#predict for the test dataset
test_transform = []
for i in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[i,2:])) #join and append all news headlines
    test_transform[i] = test_transform[i].lower() #converting to lowercase
    test_transform[i] = re.sub("[^a-zA-Z]", " ", test_transform[i]) #remove all special characters


test_dataset = cv.transform(test_transform)
#prediction
predictions = rfc_model.predict(test_dataset)

#checking the accuracies
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
confusion_m = confusion_matrix(test['Label'], predictions) #(y_true, y_predict)
score = accuracy_score(test['Label'], predictions) #(y_true, y_predict)
report = classification_report(test['Label'], predictions) #(y_true, y_predict)
