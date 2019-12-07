# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 22:27:46 2019

@ author: Hair Albeiro Parra Barrera 

This script is a wrapper of the Gensim's LDA algorithm implemented as a class  
for encapsulation and practicality purposes. 

""" 

### 1. Imports ### 

import re 
import os
import sys
import time
import argparse  
import warnings 
import pandas as pd 
import numpy as np 

import nltk
import spacy
import pickle  # used to save the model 
import gensim 

from preprocessor import nltk_preprocessor 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from gensim import corpora 
from gensim.models.ldamodel import LdaModel  


from tqdm import tqdm
from langdetect import detect


### 2. Warnings ### 

## Ignore warnings for this script ## 
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    

# English stopwords 
en_stop = set(nltk.corpus.stopwords.words('english'))

    
### 3. Command line arguments setting ### 
    
#parser = argparse.ArgumentParser(description='Parses an input text into LDA format')
#
#
#parser.add_argument("cipher_folder", help="Input cipher folder path realive to the script")
#parser.add_argument("-laplace", help="Apply Laplace smoothing", 
#                    action="store_true")
#parser.add_argument("-lm", help="Improved plaintext modelling", 
#                    action="store_true")
#
#args = parser.parse_args()

    
### 4. Class implementation ### 
    
class LDA_parser(): 
    
    def __init__(self, corpus='', language='english', preprocessor_type = "nltk", num_topics=10, passes=100): 
        """ 
        Parses the input text into a suitable format, then performs all LDA extraction tasks. 
        It expects the input corpus to be a list of texts. If input is a long string, it will attempt 
        create documents by splitting by 
        @ params: 
            @ corpus: Input corpus in str or ['str','str','str', ... ] format, where each entry
                      is a document of type str. Alternatively, a str format input (not recommended).
            @ preprocess: Removes 
        """
    
        if preprocessor_type == "nltk": 
            self.preprocessor = nltk_preprocessor(language=language)
#        if preprocessor_type == "gensim": 
#            self.preprocessor = spacy_preprocessor(language=language)
            
        self.language = language # input language 
        self.raw_corpus = "" # simply stores the input if in str type 
        self.clean_corpus = [] # [doc, doc, ..., doc] = [[sent, sent, ...], ... ,[sent, sent, ...]]
        self.dictionary = None # holds a corpora.Dictionary representation of corpus 
        self.doc2bow_corpus = None # contains doc2bow vector representations of each document in the corpus
        self.lda = None # LDA model trained on the input corpus 
        self.topic_mixtures = [] # contains str representations of mixtures of words with their probabilities  
        self.topics = {} # Contains a dictionary of topics once "extract topics" is called.  
        

        if isinstance(corpus,str): 
            print("***WARNING***\nRaw input (str) received. Text will be sentence-tokenized and parsed accordingly.")
            print("Make sure this is intended. \n")
            self.raw_corpus = str(corpus) 
            self.clean_corpus = self.preprocessor.preprocess_str_corpus(corpus)
        elif corpus == '': 
            print("***WARNING***\nNull Corpus") 
        # assume input corpus is in the right format  
        else: 
            print("Fitting LDA topic modelling...\n")
            self.fit(corpus, language=language, num_topics=num_topics, passes=passes) 
        
#        try: 
#        # Check for raw input corpus (==)
#
#                
#        except Exception as e: 
#            print("OOPS! Something went wrong, buddy...\n")
#            print("ARGUMENTS: ", e.args)
#            print("CAUSE", e.__cause__)
#            print("CONTEXT: ", e.__context__)    
#            print("SUPRESS: ", e.__suppress_context__)
#            print("TRACEBACK", e.__traceback__)
#            print(e)
#            
        return   
            
    
    def fit(self, corpus, language = 'english', num_topics=10, passes = 100, min_len=2):  
        """ 
        Assumes input corpus is in the right format. 
        @args: 
            @ corpus = input corpus  
        """
        
        self.raw_corpus = corpus # input corpus as is 
        self.language = language # in case initial language changed 
        self.clean_corpus = self.preprocessor.preprocess_texts(self.raw_corpus, min_len=2) # preprocess text list  
        self.dictionary = corpora.Dictionary(self.clean_corpus) # create corpora.Dictionary mapping 
        self.doc2bow_corpus = [dictionary.doc2bow(text) for text in self.clean_corpus] # doc2bow corpus representation 
        self.lda_model =  LdaModel(self.doc2bow_corpus, num_topics = num_topics , id2word=dictionary, passes=passes) 
        self.topic_mixtures = self.lda_model.print_topics(num_words=10) # string representation of topics mixtures   
        
        
        
    def print_topics(self, words_per_topic=5): 
        """
        Displays the topics in string format
        """ 
        topics = ldamodel.print_topics(num_words=words_per_topic) 
        for topic in topics: 
            print(topic) 
        
        
    def extract_topics(self, words_per_topic = 10, threshold = 0.005): 
        """
        Returns all topics as a dictionary of tuples, where the key is the topic 
        number, and the value is a list of words_per_topic many words with highest probability 
        composing the given
        """
        topics = {} # to store the topics 
        num_topics = len(self.topic_mixtures) # we have this many number of topics 
        
        # assign the topics mixtures  
        for i in range(num_topics): 
            topics[i] = ldamodel.show_topic(i,topn=words_per_topic)  # extract mosst probable words for topic i  
            
        self.topics = topics # update attribute  
        
        return topics 
            

    
    

        
        
# 5. TESTS  
        
PATH = "topic_modelling_dataset.xlsx"

# example df 
df = pd.read_excel(PATH) # load into a data-frame 
print(df.head()) 
print(df.columns)

text_list = list(map(str, list(df['RESULTATS_2018'])))

text_example = """This is sentence number one. 
                    This is sentence number 2. 
                    This one is number 3""" 
                    
                    
print(type(text_example))
print(isinstance(text_example,str))

parser = LDA_parser(text_list, language='french') 


### 3. Data Preprocessing ### 

from preprocessor import nltk_preprocessor 

# initialize 
cleaner = nltk_preprocessor(language='french')


# apply cleaning 
clean_texts = cleaner.preprocess_texts(text_list=text_list,
                                       lemmatize=False, 
                                       stem=False, 
                                       join=False, # generate list of tokens
                                       min_len=2) 


## LDA with Gensim ## 

# pass thte text to the corpora object and create a dictionary object  
dictionary = corpora.Dictionary(clean_texts) 
corpus = [dictionary.doc2bow(text) for text in clean_texts] 

## Save the corpus? 
#pickle.dump(corpus, open('corpus.pkl','wb')) 
#dictionary.save('dictionary.gensim') 
#

ldamodel = LdaModel(corpus, num_topics = 10 , 
                    id2word=dictionary, passes=100) 


## save the model?  
#ldamodel.save('model.gensim')


# Get the topics 
topics = ldamodel.print_topics(num_words=10)
for topic in topics: 
    print(topic) 
    

topic_words = [tup[0] for tup in ldamodel.show_topic(0,topn=10)] 
print("Example extraction words for a topic:\n", topic_words)




test_text = """C'est très difficile de faire des avances à moins qu'on commence 
                à facilitier des activités pour des enfants et les familles. Une 
                activité de plus peut faire la différence dans des projets sociaux. 
                On a donc besoin de la collaboration des organismes pour obtenir 
                des meilleurs résultats. """ 
                
     
nlp_fr = spacy.load("fr_core_news_sm")
doc = nlp_fr(test_text)


text_lemmas = cleaner.preprocess_sentence(test_text) # clean text 
new_doc_bow = dictionary.doc2bow(text_lemmas) # convert to bow dictionary 
print(ldamodel.get_document_topics(new_doc_bow)) # obtain topics  



doc_topics = ldamodel.get_document_topics(new_doc_bow) 

from operator import itemgetter

# obtain the maximally related topic 
max_topic = max(doc_topics, key=itemgetter(1))




                
                

    




 

 

        
        





