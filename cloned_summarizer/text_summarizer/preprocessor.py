# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 19:11:18 2019

@ author: Hair Albeiro Parra Barrera 

Preprocessor class for the LDA_parser class.  
"""


import re
import spacy 
from tqdm import tqdm 
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
            self.stemmer = SnowballStemmer(language=language) # Initialize Snowball stemmer 
            self.lemmatizer = WordNetLemmatizer() # Re-assign lemmatizer 
            self.stopwords = set(stopwords.words(language)) # Obtain language stopwords 
        except Exception as e: 
            print("ERROR: Invalid input language. Please check NLTK documentation for supported languages.")
            e.with_traceback() 


    def preprocess_sentence(self, 
                            sentence, 
                            custom_filter = [], 
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
            @ custom_filter: exclude a specific list of words  
            @ stem: use nltk stemmer on the tokens 
            @ lemmatize: use nltk lemmatizer on the tokens 
            @ join: if True, return the processed sentence as a str, 
                    else, return a list of processed tokens.
            @ min_len: minimum length of a token to be considered 
        """
        # Tokenize
        tokens = self.word_tokenizer(sentence) 
        
        # Remove punctuation, numbers & symbols
        tokens = [re.sub(r"[^a-zA-Z]","", token) for token in tokens ]
        
        # convert to lowercase
        tokens = [token.lower() for token in tokens]
        
        # Stem or lemmatize
        if stem: 
            tokens = [self.stemmer.stem(token) for token in tokens] 
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens] 
        
        # remove stopwords and empty strings 
        tokens = [token for token in tokens if token 
                  not in self.stopwords
                  and token not in custom_filter
                  and len(token) >= min_len] 
        
        if join: 
            return ' '.join(tokens)
        
        return tokens
        
    
    def preprocess_texts(self,
                         text_list, 
                         custom_filter = [], 
                         stem=False, 
                         lemmatize=False, 
                         join=False, 
                         min_len=1): 
        """ 
        Applies preprocess text on a list of texts. 
        
        @params: 
            @ text_list: input list of texts 
            @ custom_filter: exclude a specific list of words  
            @ stem: use nltk stemmer on the tokens 
            @ lemmatize: use nltk lemmatizer on the tokens 
            @ join: if True, return the processed sentence as a str, 
                    else, return a list of processed tokens. 
        """ 
        return [self.preprocess_sentence(text, custom_filter=custom_filter, stem=stem, lemmatize=lemmatize,
                                         join=join, min_len=min_len) for text in text_list] 
            
    
    
    def preprocess_str_corpus(self, 
                            str_corpus, 
                            custom_filter = [], 
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
        return self.preprocess_texts(documents, custom_filter=custom_filter, stem=stem, lemmatize=lemmatize, join=join, min_len=min_len)  
        
   
# ****************************************************************************** # 

class spacy_preprocessor(): 

    def __init__(self, language='english', language_model='md'): 
        """ 
        Sets up a preprocessor according to input language.  
        Please make sure the respective language models. 
        Example languages: 
            # en_core_web_sm 
            # en_core_web_md 
            # en_core_web_lg 
            # fr_core_news_sm 
            # fr_core_news_md 
            # es_core_news_sm 
            # es_core_news_md 
        @params: 
            @ language_model: spacy's language model
        
        @attributes: 
            @ word_tokenizer = nltk word tokenizer
            @ sent_tokeizer = nltk sentence tokenizer 
            @ stemmer = nltk Snowball stemmer  
            @ lemmatizer = nltk lemmatizer 
            @ stopwords =  stopwords list from input language
        """
        
        # Set preprocessor parameter according to the input language 
        # Set medium model by default if not specified. 
        if language == 'english': 
            self.language = 'english'  
            
            if language_model == 'sm': 
                self.language_model = 'en_core_web_sm' 
            elif language_model == 'lg': 
                self.language_model = 'en_core_web_lg' 
            else: 
                self.language_model = 'en_core_web_md'
                
        elif language == 'spanish':  
            self.language = 'spanish'  
            
            if language_model == 'sm': 
                self.language_model = 'es_core_news_sm' 
            else: 
                self.language_model = 'es_core_news_md' 

        elif language == 'french':
            self.language = 'french' 
            
            if language_model == 'sm': 
                self.language_model = 'fr_core_news_sm' 
            else: 
                self.language_model = 'fr_core_news_md' 

        else: 
            message = "Input language not recognized. " 
            message += "Please make sure to input a valid language."
            raise ValueError(message)
                
        try:
            
            self.nlp = spacy.load(self.language_model) # instantiate model 
            self.stopwords = set(stopwords.words(language)) # Obtain English stopwords 
            self.stemmer = SnowballStemmer(language=language) # Initialize Snowball stemmer 
            self.sent_tokenizer = sent_tokenize # Re-assign sentence tokenizer
            
        except Exception as e: 
            print("ERROR: Invalid input language. Please check NLTK documentation for supported languages.")
            print(e) 
    
    
    def preprocess_sentence(self, 
                            sentence, 
                            tags = ["DET","PUNCT","NUM","SYM","SPACE"],
                            custom_filter = [], 
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
            @ tags: filters tokens with POS_tags in input tags list
            @ custom_filter:  a list of custom words to exclude 
            @ stem: use nltk stemmer on the tokens 
            @ lemmatize: recovers lemmas using spacy's language model 
            @ join: if True, return the processed sentence as a str, 
                    else, return a list of processed tokens.
            @ min_len: minimum length of a token to be considered 
            
        @returns: 
            @ a list of clean tokens if join==false 
            @ the cleaned sentence as a string separates by spaces 
        """
        
        doc = self.nlp(sentence) # fit sentence to language model 
        
        # Tokenize, lowercase, filter numbers, stopwords, 
        # input POS_tags and tokens with length < min_len. 
        if stem: 
            tokens = [self.stemmer.stem(token.text.lower()) for token in doc 
                      if token.text.isalpha() 
                      and token.text not in self.stopwords 
                      and token.text not in custom_filter
                      and token.pos_ not in tags 
                      and len(token.text) >= min_len ]
            
        elif lemmatize: 
            tokens = [token.lemma_.lower() for token in doc 
                      if token.text.isalpha() 
                      and token.text not in self.stopwords 
                      and token.text not in custom_filter
                      and token.pos_ not in tags 
                      and len(token.text) >= min_len ]   
            
        else: 
            tokens = [token.text.lower() for token in doc 
                      if token.text.isalpha() 
                      and token.text not in self.stopwords 
                      and token.text not in custom_filter
                      and token.pos_ not in tags 
                      and len(token.text) >= min_len ]  
            
            
        if join: 
            return ' '.join(tokens)
        
        return tokens
    
    
    def preprocess_texts(self, 
                            text_list, 
                            tags = ["DET","PUNCT","NUM","SYM","SPACE"], 
                            custom_filter = [], 
                            stem=False, 
                            lemmatize=False, 
                            join=False, 
                            min_len=1): 
        """
        @params: 
            @ sentence: input sentence in str format 
            @ tags: filters tokens with POS_tags in input tags list
            @ stem: use nltk stemmer on the tokens 
            @ lemmatize: recovers lemmas using spacy's language model 
            @ join: if True, return the processed sentence as a str, 
                    else, return a list of processed tokens.
            @ min_len: minimum length of a token to be considered 
            
        @returns: 
            @ list of list of clean tokens if join== False 
            @ list of clean sentences if join == True
            
        """
        
        processed = [] 
        # apply preprocessing to sentence in the input text list, 
        # and make a list of them. 
        
        if join: 
            # make each processed sentence one string
            if stem: 
                processed = [ " ".join([self.stemmer.stem(token.text.lower()) for token in doc 
                              if token.text.isalpha() 
                              and token.text not in self.stopwords 
                              and token.text not in custom_filter
                              and token.pos_ not in tags 
                              and len(token.text) >= min_len ]) for doc in self.nlp.pipe(text_list) ] 
            elif lemmatize: 
                processed = [ " ".join([token.lemma_.lower() for token in doc 
                              if token.text.isalpha() 
                              and token.text not in self.stopwords 
                              and token.text not in custom_filter
                              and token.pos_ not in tags 
                              and len(token.text) >= min_len ]) for doc in self.nlp.pipe(text_list) ] 
            else: 
                processed = [ " ".join([token.text.lower() for token in doc 
                              if token.text.isalpha() 
                              and token.text not in self.stopwords 
                              and token.text not in custom_filter
                              and token.pos_ not in tags 
                              and len(token.text) >= min_len ]) for doc in self.nlp.pipe(text_list) ] 
                
        else: 
            # list of lists of clean tokens
            if stem: 
                processed = [ [self.stemmer.stem(token.text.lower()) for token in doc 
                              if token.text.isalpha() 
                              and token.text not in self.stopwords 
                              and token.text not in custom_filter
                              and token.pos_ not in tags 
                              and len(token.text) >= min_len ] for doc in self.nlp.pipe(text_list) ]
            elif lemmatize: 
                processed = [ [token.lemma_.lower() for token in doc 
                              if token.text.isalpha() 
                              and token.text not in self.stopwords 
                              and token.text not in custom_filter
                              and token.pos_ not in tags 
                              and len(token.text) >= min_len ] for doc in self.nlp.pipe(text_list) ]
            else: 
                processed = [ [token.text.lower() for token in doc 
                              if token.text.isalpha() 
                              and token.text not in self.stopwords 
                              and token.text not in custom_filter
                              and token.pos_ not in tags 
                              and len(token.text) >= min_len ] for doc in self.nlp.pipe(text_list) ]    
        
        return processed
    
    
    def preprocess_str_corpus(self, 
                            str_corpus, 
                            tags = ["DET","PUNCT","NUM","SYM","SPACE"], 
                            custom_filter = [], 
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
        return self.preprocess_texts(documents, 
                                     tags=tags, 
                                     custom_filter=custom_filter, 
                                     stem=stem, 
                                     lemmatize=lemmatize, 
                                     join=join,
                                     min_len=min_len)  
        
   
    
    