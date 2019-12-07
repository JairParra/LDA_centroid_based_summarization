# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:39:48 2019

@author: jairp
"""

import centroid_bow
import centroid_word_embeddings
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile # common_texts is a list 
import bs4 as BeautifulSoup # to do some basic web cleaning 
import urllib.request  


#
#print(common_texts[0:2]) # all of these texts are pre-processed?  



# Test centroid model by summerizing wikipedia pages

while True:
    topic = input('What topic to summarize? ').replace(' ', '_').lower()
    
    #fetching the content from the URL
    fetched_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/' + topic)
    
    # reading the article 
    article_read = fetched_data.read()

    #parsing the URL content and storing in a variable
    article_parsed = BeautifulSoup.BeautifulSoup(article_read,'html.parser')

    #returning <p> tags
    paragraphs = article_parsed.find_all('p')

    # to store the article content      
    article_content = ''

    #looping through the paragraphs and adding them to the variable
    for p in paragraphs:  
        article_content += p.text


    # construct the model with texts    
    model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=-1) 
    sume = centroid_word_embeddings.CentroidWordEmbeddingsSummarizer(model)
    print(sume.summarize(article_content))
    
    
