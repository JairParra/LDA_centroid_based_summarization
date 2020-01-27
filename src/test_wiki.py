# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:39:48 2019

@author: Logan Raltson and Hair Parra
"""

import re
import centroid_bow
import centroid_word_embeddings
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile # common_texts is a list 
from LDA_extractor import LDA_parser
from preprocessor import spacy_preprocessor 
import bs4 as BeautifulSoup # to do some basic web cleaning 
import urllib.request  


topic = "computer_science" # default

# Test centroid model by summarizing wikipedia pages
while topic is not None: 
    
    # obtain the topic 
    topic = input('What topic to summarize? ').replace(' ', '_').lower()
    
    if re.search(r'no|No|exit',topic, flags=re.IGNORECASE):  
        break
        
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

    # construct Word2Vec word embeddings model
    model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=-1) 
    
    # instantiate custom parser 
    parser = LDA_parser(article_content, 
                    language='english', 
                    preprocessor_type='spacy', 
                    num_topics = 50, 
                    passes = 100, 
                    min_len=2  # min len of words to be considered 
                    ) 
    
    sumarizer = centroid_word_embeddings.CentroidWordEmbeddingsSummarizer(model, parser=parser)
    print("SUMMARY:\n", sumarizer.summarize(article_content))
    
    
    
#### TESTS 
#topic = "nintendo"
#
## fetch data from url
#fetched_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/' + topic) 
#
## reading the article 
#article_read = fetched_data.read()
#
##parsing the URL content and storing in a variable
#article_parsed = BeautifulSoup.BeautifulSoup(article_read,'html.parser')
#
##returning <p> tags
#paragraphs = article_parsed.find_all('p')
#
## to store the article content      
#article_content = ''
#
##looping through the paragraphs and adding them to the variable
#for p in paragraphs:  
#    article_content += p.text
#
## examine article 
#print(type(article_content)) 
#
### Text cleaner 
##cleaner = spacy_preprocessor(language='english')
##
### obtain sentences with preprocessor 
##sents = cleaner.preprocess_str_corpus(article_content)
#
#
#
## prepprocess 
#parser = LDA_parser(article_content, 
#                language='english', 
#                preprocessor_type='spacy', 
#                num_topics = 50, 
#                passes = 100, 
#                min_len=2  # min len of words to be considered 
#                ) 
#
#summarizer = centroid_word_embeddings.CentroidWordEmbeddingsSummarizer(model, parser=parser)
#print(sumarizer.summarize(article_content))










