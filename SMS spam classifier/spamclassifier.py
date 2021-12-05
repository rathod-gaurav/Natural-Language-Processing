# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 01:41:25 2021

@author: ratho
"""
import pandas as pd
messages = pd.read_csv('spamSMS', sep='\t', names=["label", "message"])

import re
import nltk 
#nltk.download()
#nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
lt = WordNetLemmatizer()
corpus = []

for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i]) #remove all special characters except 'a-z' and 'A-Z'
    review = review.lower() #convert all messages to lowercase
    review = review.split() #split the message into words
    
    #now we apply stemming process to each word in the message and remove the stopwords
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review) #join the list of all words after stemming and removing stopwords
    corpus.append(review)

#creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray() #creates a matrix of scores

#convert our ham and spam values in 'label' column into dummy variables
y = pd.get_dummies(messages['label']) #gives two categoral features 'ham' & 'spam' to each message
y=y.iloc[:,1].values #removing one categoral feature and using only one column to know whether a message is 'ham' or 'spam' 

#now, we have our dependent feature (X) and independent feature (y)

#doing a train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#training the model using naive-bayes algorithm
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train) #our model will get created

#predicting on test data based on the trained model
y_pred=spam_detect_model.predict(X_test)

#now we will compare the results (y_pred) with y_test using confusion matrix
from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test,y_pred)

#checking the accuracies
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)


    
    