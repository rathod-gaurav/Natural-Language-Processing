# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:36:25 2021

@author: ratho
"""

#Load nlp packages
import spacy
nlp = spacy.load('en_core_web_md')

#spacy.__version__

#explore components of NLP pipeline
nlp.components
nlp.pipeline 
#Get only component names
nlp.component_names
nlp.pipe_names

# Using SpacyTextBlob for sentiment analysis using TextBlob
from spacytextblob.spacytextblob import SpacyTextBlob

#adding SpacyTextBlob to NLP pipeline
nlp.add_pipe('spacytextblob')

#again view pipeline components
nlp.component_names
nlp.pipe_names

#sentiment analysis
mytext = "I love eating apples when I work at google"

docx = nlp(mytext)

#sentiment polarity and subjectivity
docx._.polarity
docx._.subjectivity

#check assessment: list polarity/subjectivity for the assessed token
docx._.assessments

#Basics
for token in docx:
    print(token.text, token.pos_, token.tag_)