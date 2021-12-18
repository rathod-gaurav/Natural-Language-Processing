# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 02:06:40 2021

@author: ratho
"""

import spacy
import pandas as pd

#loading the language model
nlp = spacy.load("en_core_web_md")

with open('trump_speech.txt') as f:
    speech = f.read().strip() #read the file and remove any spaces at start and end of file.

#identifying named entities using SpaCy model
speech_identify = nlp(speech)

#storing the named entity categories (labels) in the 'speech_identify' in a list
labels = [x.label_ for x in speech_identify.ents] #the named entities are themselves stored in speech_identify.ents

#for i in range(len(speech_identify.ents)):
#    print (speech_identify.ents[i], " : ", labels[i], sep="")

#creating a dataframe and exporting output as a csv file
data = {'Named Entity' : speech_identify.ents,
        'POS label' : labels}

df = pd.DataFrame(data)
df.to_csv('out.csv', index=False)