"""
Create LDA model using of DUC-2004 task 2 dataset
"""
import pickle
import LDA_extractor

# Load pickle containing DUC corpus
corpus = {}
with open('corpus.pkl', 'rb') as f:
        corpus = pickle.load(f)

# Create list of every article to train lda on
articles = []
for set_id in corpus.keys():
    articles = articles + corpus[set_id]['articles']

# Train LDA model on articles
parser = LDA_extractor.LDA_parser(articles,
                                  language='english', 
                                  preprocessor_type='spacy',
                                  num_topics = 100, 
                                  passes = 100)

parser.print_topics(words_per_topic = 10) 
topic_mixtures = parser.extract_topics(max_words_per_topic=50, threshold=0.005)
print(topic_mixtures)

# extract topics as a fictionary 
topics = parser.extract_topic_words(max_words_per_topic=50, threshold=0.005)
print(topics)


# Pickle LDA model
with open('model.pkl', 'wb') as output_file:
    pickle.dump(parser, output_file, pickle.HIGHEST_PROTOCOL)

# Load pickle containing model
# parser = None
# with open('model.pkl', 'rb') as f:
#        parser = pickle.load(f)

# Map one set of articles to a big compound articles
compound_article =  ''.join(corpus[list(corpus.keys())[1]]['articles'])

max_topic, doc_max_topic_words, doc_topics, doc_topic_words = parser.parse_new(compound_article)
