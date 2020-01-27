"""
Implementation based on paper:

Centroid-based Text Summarization through Compositionality of Word Embeddings

Author: Gaetano Rossiello
Email: gaetano.rossiello@uniba.it
"""
import base
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.models import KeyedVectors  # Used to load the pre-trained word embeddings 
import gensim.downloader as gensim_data_downloader
import random

def average_score(scores):
    score = 0
    count = 0
    for s in scores:
        if s > 0:
            score += s
            count += 1
    if count > 0:
        score /= count
        return score
    else:
        return 0


def stanford_cerainty_factor(scores):
    score = 0
    minim = 100000
    for s in scores:
        score += s
        if s < minim & s > 0:
            minim = s
    score /= (1 - minim)
    return score


def get_max_length(sentences):
    max_length = 0
    for s in sentences:
        l = len(s.split())
        if l > max_length:
            max_length = l
    return max_length


def load_gensim_embedding_model(model_name):
    """ 
    Function to load and select the word embeddings model, using the Gensim lirbary. 
    """
    available_models = gensim_data_downloader.info()['models'].keys()
    assert model_name in available_models, 'Invalid model_name: {}. Choose one from {}'.format(model_name, ', '.join(available_models))
    model_path = gensim_data_downloader.load(model_name, return_path=True)
    return KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')


class CentroidWordEmbeddingsSummarizer(base.BaseSummarizer):
    """
    Extends the previous BaseSummarizer class which contains basic 
    fitting attributes and preprocessing methods.  
    This class implements the summarize function 
    Word embeddings are obtained through Gensim, instead of BOW vectors (i.e. CountVectorizer methods)
    """ 
    
    def __init__(self,
                 embedding_model, # word embeddings 
                 language='english',  
                 preprocess_type='nltk',
                 stopwords_remove=True, 
                 length_limit=10, # min length limit for sentences when sent_tokenizing in preprocessing step 
                 debug=False,
                 topic_threshold=0.3,
                 sim_threshold=0.93, 
                 reordering=True,  
                 zero_center_embeddings=False, # re-center embeddings to have zero cetner  
                 keep_first=False, # ??? 
                 bow_param=0, 
                 length_param=0,
                 position_param=0,
                 parser = None):
        """ 
        This is the Centroid method based on word2vec embedding representations.
        
        @attributes: 
            @ self.topic_threshold: Select "meaningful" words in the document to form the centroid 
              embeddings, which have a tfidf score greater than topic threshold. 
            @ self.sim_threshold: Discard sentences that are too similar to avoid redundancy 
        """
        
        # initalize base parent class with preprocessing parameters 
        super().__init__(language, preprocess_type, stopwords_remove, length_limit, debug)

        self.embedding_model = embedding_model  # input word embedding modele 
        #print(self.embedding_model['said'])
        self.vector_model = self.embedding_model.wv
        self.word_vectors = dict() # initialize dictionary to store word vectors 

        self.topic_threshold = topic_threshold  # select words with tfidf score greater than threshold to compose the centroid
        self.sim_threshold = sim_threshold # avoid sentences that are too similar 
        self.reordering = reordering

        self.keep_first = keep_first
        self.bow_param = bow_param # ??? 
        self.length_param = length_param # ??? 
        self.position_param = position_param # ???
        self.parser = parser

        self.zero_center_embeddings = zero_center_embeddings

        if zero_center_embeddings:
            self._zero_center_embedding_coordinates()
        return

    def get_bow(self, sentences):
        """ 
        Obtain a BOW vector representation of the input sentences. 
        It uses the eaxact same technique as the cnetroid_bow class. 
        """
        vectorizer = CountVectorizer() # instantiate CountVectorizer object 
        sent_word_matrix = vectorizer.fit_transform(sentences)# create mapping of sentences  

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False) # instantiate tfidf weighting  
        tfidf = transformer.fit_transform(sent_word_matrix) # fit tfidf weighting to the counts matrix 
        tfidf = tfidf.toarray() # convert to dense ndarray

        centroid_vector = tfidf.sum(0) # sum of all tfidf vectors to create the centroid  (axis=0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max()) # normalization  
        for i in range(centroid_vector.shape[0]):  # for every entry in the vector  
            if centroid_vector[i] <= self.topic_threshold: # if entry is not relevant , set it to 0 
                centroid_vector[i] = 0
        return tfidf, centroid_vector # return tfidf vector and the centroid vector from the tfidf vectors 

    def get_topic_idf(self, sentences):
        """
        Extracts all the words from the centroid based on the input sentences, whenever their 
        aggregated tfidf is more thatn a certain topic threshold. 
        """
        n_most_likely_words, top_n_topics, top_n_words, doc_topics, doc_topic_words = self.parser.parse_new(' '.join(sentences), top_n_w=30, verbose = False)
        #print(max_topic)
        
        #vectorizer = CountVectorizer() # instantiate COuntVectorizer object 
        #sent_word_matrix = vectorizer.fit_transform(sentences) # fit the input sentences 

        #transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False) # instantiate tfidf weighting  
        #tfidf = transformer.fit_transform(sent_word_matrix) # fit the BOW representation of matrix 
        #tfidf = tfidf.toarray() # convert to array 

        #centroid_vector = tfidf.sum(0)  # sum all vectors into one 
        #centroid_vector = np.divide(centroid_vector, centroid_vector.max())  # normalize 
        # print(centroid_vector.max())

        #feature_names = vectorizer.get_feature_names() # obtain vocaulary list  

        # extract words whose tfidf has relevance based on the topic_threshold parameter
        #relevant_vector_indices = np.where(centroid_vector > self.topic_threshold)[0] 

        #word_list = list(np.array(feature_names)[relevant_vector_indices]) # obtain a list of such words

        return n_most_likely_words

    def word_vectors_cache(self, sentences):
        """  
        Fills the word_vectors attribute of the class by obtaining the respective vectors 
        from the word embedding model or a centered version of it using self.centroid_space 
        """
        self.word_vectors = dict()
        for s in sentences: # for each sentence 
            words = s.split()  # split by word s
            for w in words: # for each word
                if w in self.vector_model.vocab: # if the vector for that word exists
                    if self.zero_center_embeddings: # if zero center embeddings acivated 
                        self.word_vectors[w] = (self.embedding_model[w] - self.centroid_space) # replace
                    else:
                        #print(w)
                        self.word_vectors[w] = self.embedding_model[w]
        #print("Final cache:")
        #print(self.word_vectors.keys())
        return

    # Sentence representation with sum of word vectors
    def compose_vectors(self, words, debug=False):
        """ 
        This function obtains the sentence representation of the input words  
        by aggregating their vector representations and composing them. 
        """
        composed_vector = np.zeros(self.embedding_model.vector_size, dtype="float32") # initialize vector of zeros 
        word_vectors_keys = set(self.word_vectors.keys()) # 
        count = 0
        #print('word vector keys')
        #print(word_vectors_keys)
        for w in words:
            if w in word_vectors_keys: # for each word in the vectors of the model 
                #print("a")
                #print(self.word_vectors[w])
                composed_vector = composed_vector + self.word_vectors[w] # add the word vector to the overall vector
                count += 1
        if count != 0:
            composed_vector = np.divide(composed_vector, count) # normalize 
        return composed_vector

    def summarize(self, text, limit_type='word', limit=100):
        """ 
        Main function for text summarization using word2vec embeddings. 
        The idea is similar to the previous , but more extense. 
        """
        raw_sentences = self.sent_tokenize(text) # tokenize by sentences
        clean_sentences = self.preprocess_text(text)  # preprocess sentences with model's prepreocessor (nltk or regex)

        if self.debug:
            print("ORIGINAL TEXT STATS = {0} chars, {1} words, {2} sentences".format(len(text),
                                                                                     len(text.split(' ')),
                                                                                     len(raw_sentences)))
            print("*** RAW SENTENCES ***")
            for i, s in enumerate(raw_sentences):
                print(i, s)
            print("*** CLEAN SENTENCES ***")
            for i, s in enumerate(clean_sentences):
                print(i, s)

        centroid_words = self.get_topic_idf(clean_sentences) # obtain all word whose tfidf id more than topic_threshold
        #print(centroid_words)
        
        if self.debug:
            print("*** CENTROID WORDS ***")
            print(len(centroid_words), centroid_words)

        self.word_vectors_cache(clean_sentences)  # obtain word representations from the model embeddings  
        centroid_vector = self.compose_vectors(centroid_words, debug=True) # obtain centroid vector from the centroid word embeddings 

        tfidf, centroid_bow = self.get_bow(clean_sentences) # obtain tfidf vectors ebmeddings and the resulting tfidf vector 
        max_length = get_max_length(clean_sentences) # max lenght of the clean sentences  

        sentences_scores = [] # to store the scores  
        for i in range(len(clean_sentences)): # for each sentence in clean sentences 
            scores = []
            words = clean_sentences[i].split()  # obtain agggregates words 
            sentence_vector = self.compose_vectors(words) # obtain the sentence embedding representation 

            scores.append(base.similarity(sentence_vector, centroid_vector)) # obtain similarity between sentence and centroid
            scores.append(self.bow_param * base.similarity(tfidf[i, :], centroid_bow)) # obtain similarity score of the ith tfidf word vector and centroid 
            scores.append(self.length_param * (1 - (len(words) / max_length))) # score for the length and complement ration of the numer of words and sentence max lentgth
            scores.append(self.position_param * (1 / (i + 1)))  # score for relative position of the sentence (position matters in relevance)
            score = average_score(scores) # average over all the scores obtained 
            # score = stanford_cerainty_factor(scores)
            
            sentences_scores.append((i, raw_sentences[i], score, sentence_vector))  # (index, sentence i, avg score, sentence vector)

            if self.debug:
                print(i, scores, score)

        sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True) # sort the sentences by their relative score 
        if self.debug:
            print("*** SENTENCE SCORES ***")
            for s in sentence_scores_sort:
                print(s[0], s[1], s[2])

        count = 0
        sentences_summary = []

        if self.keep_first: # fi we want to keep the first sentence 
            for s in sentence_scores_sort:
                if s[0] == 0:
                    sentences_summary.append(s)
                    if limit_type == 'word':
                        count += len(s[1].split())
                    else:
                        count += len(s[1])
                    sentence_scores_sort.remove(s)
                    break

        for s in sentence_scores_sort: # for each sentence in sorted sentences 
            if count > limit: # if the count exceeeds limit, break the algorihtm 
                break
            include_flag = True
            for ps in sentences_summary: # for every other sentence 
                sim = base.similarity(s[3], ps[3])  #somare how similar they are 
                # print(s[0], ps[0], sim)
                if sim > self.sim_threshold:  # if too similar 
                    include_flag = False  # don't include
            if include_flag:
                # print(s[0], s[1])
                sentences_summary.append(s) # append 
                if limit_type == 'word': # decide limit by word or by sentence 
                    count += len(s[1].split()) 
                else:
                    count += len(s[1])

        if self.reordering: # reorderby sorting 
            sentences_summary = sorted(sentences_summary, key=lambda el: el[0], reverse=False)

        summary = "\n".join([s[1] for s in sentences_summary]) # obtain summary by joinin the sentences  

        if self.debug:
            print("SUMMARY TEXT STATS = {0} chars, {1} words, {2} sentences".format(len(summary),
                                                                                    len(summary.split(' ')),
                                                                                    len(sentences_summary)))

            print("*** SUMMARY ***")
            print(summary)

        return summary

    def _zero_center_embedding_coordinates(self):
        # Create the centroid vector of the whole vector space
        count = 0
        self.centroid_space = np.zeros(self.embedding_model.vector_size, dtype="float32") #create a vector of zeros 
        self.index2word_set = set(self.embedding_model.wv.index2word) # class model's vectors index2word unique mapping 
        for w in self.index2word_set: # for every word in this mapping  
            self.centroid_space = self.centroid_space + self.embedding_model[w] # add the vector from the model's embedding 
            count += 1 # increment count 
        if count != 0: 
            self.centroid_space = np.divide(self.centroid_space, count) # normalize 
