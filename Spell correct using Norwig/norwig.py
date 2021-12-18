# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 03:07:08 2021

@author: ratho
"""

import re
from collections import Counter

def words(text):
    return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('dictionary.txt').read()))

#check the probability of a given sentence
def check(sentence):
    sentence = sentence.lower()
    sentence = sentence.split()
    probabilities = []
    probability = [P(max_prob_word(word)) for word in sentence]
    probabilities.append(probability)
    return sum(probabilities)/len(sentence)
    
    
#all edits that are one distance away from word
def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L,R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L,R in splits if len(R)>1]
    replaces = [L + c + R[1:] for L,R in splits if R for c in letters]
    inserts = [L + c + R for L,R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

#all edits that are two distance away from word
def edits2(word):
    return (w2 for w1 in edits1(word) for w2 in edits1(w1))

#set of words which are known to us
def known(words):
    return set(w for w in words if w in WORDS)

#generate possible spelling corrections for the given word
def possible_corr(word):
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

#probability of 'word'
import numpy as np
def P(word, N=sum(np.hstack(WORDS.values()))):
    return WORDS[word]/N

#find maximum probable word from all corrections suggested
def max_prob_word(word):
    return max(possible_corr(word), key=P)


check('hello world')






