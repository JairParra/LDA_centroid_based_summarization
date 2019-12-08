"""
     Author: Gaetano Rossiello
     Email: gaetano.rossiello@uniba.it
"""
import base
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class CentroidBOWSummarizer(base.BaseSummarizer):
    """
    Extends the previous BaseSummarizer class which contains basic 
    fitting attributes and preprocessing methods.  
    This class implements the "summarize function" 
    
    """ 

    def __init__(self,
                 language='english',
                 preprocess_type='nltk',
                 stopwords_remove=True,
                 length_limit=10,
                 debug=False,
                 topic_threshold=0.3,
                 sim_threshold=0.95):
        """ 
        This is the Centroid method based on BOW word vector representations. 
        This means that the vectors produced are one-hot-encoding representations, 
        base on the input sentences. 
        
        @attributes: 
            @ self.topic_threshold: Select "meaningful" words in the document to form the centroid 
              embeddings, which have a tfidf score greater than topic threshold. 
            @ self.sim_threshold: Discard sentences that are too similar to avoid redundancy 
        """
        super().__init__(language, preprocess_type, stopwords_remove, length_limit, debug)
        self.topic_threshold = topic_threshold # minimum tfidf score for words to compose centroid embedding
        self.sim_threshold = sim_threshold # smilarity threshold? 
        return

    def summarize(self, text, limit_type='word', limit=100):
        
        raw_sentences = self.sent_tokenize(text) # sent_tokenize input text
        clean_sentences = self.preprocess_text(text) # preprocess sentences using nltk ot regex 

        vectorizer = CountVectorizer() # instantiate CountVectorizer object 
        sent_word_matrix = vectorizer.fit_transform(clean_sentences) # create mapping of sentences  

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False) # instantiate tfidf weighting  
        tfidf = transformer.fit_transform(sent_word_matrix) # fit tfidf weighting to the counts matrix 
        tfidf = tfidf.toarray() # convert to numpy array 

        centroid_vector = tfidf.sum(0) # centroid vector for the input text 
        centroid_vector = np.divide(centroid_vector, centroid_vector.max()) # normalize by the maximum value 
        for i in range(centroid_vector.shape[0]): # for i=0 to m (sentences)
            if centroid_vector[i] <= self.topic_threshold:
                centroid_vector[i] = 0 


        sentences_scores = [] # store sentence scores  
        for i in range(tfidf.shape[0]): 
            score = base.similarity(tfidf[i, :], centroid_vector) # compute similarity of the sentences and centroid vector 
            sentences_scores.append((i, raw_sentences[i], score, tfidf[i, :])) # (i, sentence i, score , tfidf_vector)
            
        sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)  # DESC sort sentence scores 

        count = 0
        sentences_summary = [] # to store the summary  
        for s in sentence_scores_sort:
            # summary is commposed of limit number of sentences 
            if count > limit: 
                break
            include_flag = True # include the sentence or not  
            for ps in sentences_summary: # for each sentence i the current summary 
                sim = base.similarity(s[3], ps[3]) # obtain their similarity scores  
                # print(s[0], ps[0], sim)
                if sim > self.sim_threshold: # if too similar to existent sentence in summary, discard 
                    include_flag = False
            if include_flag: 
                # print(s[0], s[1])
                sentences_summary.append(s) # include sentence inthe summary  
                if limit_type == 'word': 
                    count += len(s[1].split()) # max summary length is total number of words
                else:
                    count += len(s[1]) # max summary legnth is total number of sentences 

        summary = "\n".join([s[1] for s in sentences_summary]) # create the summary. 
        return summary
