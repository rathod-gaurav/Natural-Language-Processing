# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 00:58:47 2021

@author: ratho
"""
from os import listdir
from nltk.corpus import stopwords
import string
from collections import Counter

import numpy as np

#load doc into memory
def load_doc(filename):
    file = open(filename,'r')
    text = file.read()
    file.close()
    return text

#clean text
def clean_doc(doc):
    tokens = doc.split() #split into tokens by white space
    table = str.maketrans('', '', string.punctuation) #remove punctuation from each token
    tokens = [word.translate(table) for word in tokens]
    tokens = [word for word in tokens if word.isalpha()] #remove non-alphabetic tokens
    stop_words = set(stopwords.words('english')) #remove stopwords
    tokens = [word for word in tokens if not word in stop_words]
    tokens = [word for word in tokens if len(word)>1] #filter out short tokens
    return tokens

#load doc and add it to the vocabulary
def add_doc_to_vocab(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)

#load all docs in a directory
def process_docs(directory, vocab, is_train):
    #see all docs in the folder
    for filename in listdir(directory):
        #skip any reviews in the test set
        if is_train and filename.startswith('cv9'):
            continue
        #creating path of filename to open
        path = directory + '/' + filename
        #add doc to vocab
        add_doc_to_vocab(path, vocab)

#define vocabulary
vocab = Counter()

#add all docs to vocab
process_docs('txt_sentoken/neg', vocab, True)
process_docs('txt_sentoken/pos', vocab, True)

#print(len(vocab)) #size of vocabulary
#print(vocab.most_common(50)) #print top 50 most common words in vocab

#keep tokens only with a minimum occurence
min_occurence = 2
tokens = [k for k,c in vocab.items() if c>=min_occurence]
#print(len(tokens))

#saving the vocabulary to a new file "vocab.txt"
def save_list(lines, filename):
    data = '\n'.join(lines) #join all lines
    file=open(filename, 'w') #open file
    file.write(data) #write text into the file
    file.close() #close file

save_list(tokens, 'vocab.txt') #save tokens to a vocab.txt file

########################################
########################################
####### Training embedding layer #######
########################################
########################################

#loading the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename) #vocab is of dtype str
vocab = vocab.split() #vocab is of dtype list
vocab = set(vocab) #vocab is of dtype set

#cleaning the document
def clean_doc_updated(doc, vocab):
    tokens = doc.split() #split tokens by white space
    table = str.maketrans('', '', string.punctuation) #remove punctuation from each token
    tokens = [word.translate(table) for word in tokens]
    tokens = [word for word in tokens if word in vocab] #filter out only those words in the tokens which are present in our vocabulary
    tokens = ' '.join(tokens) #joining all tokens to return a list of cleaned string
    return tokens

#loading and processing all the docs in the directory
def process_docs_updated(directory, vocab, is_train):
    documents = list() #list of all strings generated from clean_doc_updated for each filename in the 'pos' and 'nog' folders
    #see all files in the folder
    for filename in listdir(directory):
        if is_train and filename.startswith('cv9'): #skip test set
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        #create full path of filename to open
        path = directory + '/' + filename
        #loading the doc
        doc = load_doc(path)
        #cleaning the doc
        tokens = clean_doc_updated(doc, vocab)
        #add tokens to documents list
        documents.append(tokens)
    return documents

#load all training reviews
positive_docs = process_docs_updated('txt_sentoken/pos', vocab, True)
negative_docs = process_docs_updated('txt_sentoken/neg', vocab, True)
train_docs = negative_docs + positive_docs

##############################################################
###### Encoding each document as a sequence of integers ######
##############################################################

#It develops a vocabulary of all tokens in the training dataset 
#and develops a consistent mapping from words in the vocabulary to unique integers
#from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#creating the tokenizer
tokenizer = Tokenizer()
#fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

#encoding rewiews in the training dataset
encoded_docs = tokenizer.texts_to_sequences(train_docs)

#padding all reviews to the length of longest review in the training dataset
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

#define training labels - 0 for negative reviews and 1 for positive reviews
ytrain = np.array([0 for _ in range(900)] + [1 for _ in range(900)])


############### Encoding and Padding Test dataset ############
test_positive_docs = process_docs_updated('txt_sentoken/pos', vocab, False)
test_negative_docs = process_docs_updated('txt_sentoken/neg', vocab, False)
test_docs = test_negative_docs + test_positive_docs

#tokenizer.fit_on_texts(test_docs) #this updates internal vocabulary based on a list of texts, so we will not be executing it here
test_encoded_docs = tokenizer.texts_to_sequences(test_docs) #sequence encode
#padding sequences
Xtest = pad_sequences(test_encoded_docs, maxlen=max_length, padding='post')
#defining test labes - 0 for negative reviews and 1 for positive reviews
ytest = np.array([0 for _ in range(100)] + [1 for _ in range(100)])


########################################################
########### Defining Neural Network Model ##############
########################################################

#The Embedding requires the specification of the vocabulary size, the size of the real-valued vector space, and the maximum length of input documents.

#defining vocabulary size
vocab_size = len(tokenizer.word_index) + 1 #The vocabulary size is the total number of words in our vocabulary, plus one for unknown words

#size of vector space = 100
vs_size = 100

#maxmum length of input documents = max_length

#define model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Embedding(vocab_size, vs_size, input_length=max_length)) #Embedding layer
model.add(Conv1D(filters=32, kernel_size=8, activation='relu')) #CNN layer
model.add(MaxPooling1D(pool_size=2)) #Pooling layer to reduce the output of convolutional layer by half

model.add(Flatten()) #flatten layer to flatten the 2D output of CNN layer to one long 2D vector (this represents the features extracted by CNN)
model.add(Dense(10, activation='relu')) #Output layers
model.add(Dense(1, activation='sigmoid')) 

#print(model.summary())


############## fitting the network on training data ##############

#compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)


###############################################################
############ Evaluating the model on test dataset #############
###############################################################
loss,acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))

#predict sentiment scores of reviews in test dataset
ypred = model.predict(Xtest)


























































































