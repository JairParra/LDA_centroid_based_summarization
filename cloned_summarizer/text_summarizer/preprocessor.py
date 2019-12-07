# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 19:11:18 2019

@ author: Hair Albeiro Parra Barrera 

Preprocessor class for the LDA_parser class.  
"""


import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer


class nltk_preprocessor(): 
    
    def __init__(self, language='english'): 
        """ 
        Sets up a preprocessor according to input language. 
        Example languages: 'english','french','spanish'
        @attributes: 
            @ word_tokenizer = nltk word tokenizer
            @ sent_tokeizer = nltk sentence tokenizer 
            @ stemmer = nltk Snowball stemmer  
            @ lemmatizer = nltk lemmatizer 
            @ stopwords =  stopwords list from input language
        """
        
        try: 
            self.word_tokenizer = word_tokenize # Re-assign word tokenizer 
            self.sent_tokenizer = sent_tokenize # Re-assign sentence tokenizer
            self.stemmer = SnowballStemmer(language=language) # Initialize English Snowball stemmer 
            self.lemmatizer = WordNetLemmatizer() # Re-assign lemmatizer 
            self.stopwords = set(stopwords.words(language)) # Obtain English stopwords 
        except Exception as e: 
            print("ERROR: Invalid input language. Please check NLTK documentation for supported languages.")
            e.with_traceback() 


    def preprocess_sentence(self, 
                            sentence, 
                            stem=False, 
                            lemmatize=False, 
                            join=False, 
                            min_len=1): 
        """
        Cleans text list by applying the following steps: 
            1. Tokenize the input sentence 
            2. Remove punctuation, symbols and unwanted characters
            3. Convert the tokens to lowercase 
            4. Stem or lemmatize (according to input)
            5. Remove stopwords and empty strings
            
        @params: 
            @ sentence: input sentence in str format 
            @ stem: use nltk stemmer on the tokens 
            @ lemmatize: use nltk lemmatizer on the tokens 
            @ join: if True, return the processed sentence as a str, 
                    else, return a list of processed tokens.
            @ min_len: minimum length of a token to be considered 
        """
        # Tokenize
        tokens = self.word_tokenizer(sentence) 
        
        # Remove punctuation & symbols
        tokens = [re.sub(r"[^a-zA-Z]","", token) for token in tokens ]
        
        # convert to lowercase 
        tokens = [token.lower() for token in tokens]
        
        # Stem or lemmatize
        if stem: 
            tokens = [self.stemmer.stem(token) for token in tokens] 
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens] 
        
        # remove stopwords and empty strings 
        tokens = [token for token in tokens if token not in self.stopwords
                  and len(token) >= min_len] 
        
        if join: 
            return ' '.join(tokens)
        
        return tokens
        
    
    def preprocess_texts(self,
                         text_list, 
                         stem=False, 
                         lemmatize=False, 
                         join=False, 
                         min_len=1): 
        """ 
        Applies preprocess text on a list of texts. 
        
        @params: 
            @ text_list: input list of texts 
            @ stem: use nltk stemmer on the tokens 
            @ lemmatize: use nltk lemmatizer on the tokens 
            @ join: if True, return the processed sentence as a str, 
                    else, return a list of processed tokens. 
        """ 
        return [self.preprocess_sentence(text, stem=stem, lemmatize=lemmatize,
                                         join=join, min_len=min_len) for text in text_list] 
            
    
    
    def preprocess_str_corpus(self, 
                            str_corpus, 
                            stem=False, 
                            lemmatize=False, 
                            join=False, 
                            min_len=1):  
        """
        Input text is a long string. The parser will attempt to split by sentence, 
        and the process these sentences accordingly. 
            
        @params: 
            @ str_corpus: input corpus in str format 
            @ stem: use nltk stemmer on the tokens 
            @ lemmatize: use nltk lemmatizer on the tokens 
            @ join: if True, return the processed sentence as a str, 
                    else, return a list of processed tokens.
            @ min_len: minimum length of a token to be considered 
        """       
        
        documents = self.sent_tokenizer(str_corpus) # tokenize by sentences, result is a list. 
        return self.preprocess_texts(documents, stem=stem, lemmatize=lemmatize, join=join, min_len=min_len)  
        
        
    

        
        
    
    
    
    
    
    
    