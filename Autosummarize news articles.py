#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:37:58 2019

@author: Loonycorn
"""

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest
from bs4 import BeautifulSoup
import requests

def get_only_text_TheWire(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, features="html.parser")
    text = ' '.join(map(lambda p: p.text, soup.find_all('p'))) #mushes together all paras
    return text

url = 'https://thewire.in/sadak-se-sansad/how-smart-city-bhubaneswar-has-endangered-a-river-and-many-lives/kpDXOJW8cOw'
textOfURL = get_only_text_TheWire(url)
sents = list(set(sent_tokenize(textOfURL)))
words_per_sent =  [word_tokenize(s.lower()) for s in sents]
CustomStopwords = set(stopwords.words('english') + list(punctuation))

min_cut= 0.1
max_cut= 0.9
freq = defaultdict(int)
for sentence in words_per_sent:
    for word in sentence:
        if word not in CustomStopwords:
            freq[word] += 1

# normalising frequencies makes words easier to compare
max_freq = float(max(freq.values()))

for word in list(freq):
    freq[word] = freq[word]/max_freq
    if freq[word] >= max_cut or freq[word] <= min_cut:
        del freq[word]

rankint = defaultdict(int)

for i, sent in enumerate(words_per_sent):
    for word in sent:
        if word in freq:
            rankint[i] += freq[word]

sents_idx = nlargest(n= 3, iterable= rankint, key= rankint.get)
for index in sents_idx:
    print(sents[index])
    print()