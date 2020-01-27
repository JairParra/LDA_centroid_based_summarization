[![Generic badge](https://img.shields.io/badge/Improving_Text_Summarization_Through_LDA-blue.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Contributors-3-<COLOR>.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/COMP550-Natural_Language_Processing-red.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Status-Building.svg)](https://shields.io/)

# Improving Centroid-Based Text Summarization through LDA-based Document Centroids 

Automatic text summarization is the task of producing a text summary "from one or more texts, that conveys important information in the original text(s), and that is no longer than half of the original text(s) and usually, significantly less than that" \cite{summarization}.(Dragomir  R  Radev  and  McK-eown,  2002). 
We adapt a recent centroid-based text summarization model, one that takes advantage of the compositionality of word embeddings, in order to obtain a single vector representation of the most meaningful words in a given text. We propose utilizing Latent Dirichlet Allocation (LDA), a probabilistic generative model for collections of discrete data, in order to better obtain the topic words of a document for use in constructing the centroid vector. We see that the LDA implementation results in overall more coherent summaries, suggesting the potential for utilizing topic models to improve upon the general centroid-based method.  

## Our paper: 
- https://drive.google.com/file/d/1plsPxIYHsWAtW50tYm7YJvOgpuveqFqn/view?usp=sharing

## This work is based on: 
- **Centroid-based Text Summarization through Compositionality of Word Embeddings** https://www.aclweb.org/anthology/W17-1003.pdf 
- **Repo:** https://github.com/gaetangate/text-summarizer 

## Running the Code
1. Download the Google Vectors from https://github.com/mmihaltz/word2vec-GoogleNews-vectors and place them into the `data_clean` folder. 
2. Copy all directories from `duc2004\testdata\tasks1and2\t1.2\docs` (DUC data not distributed in this repo due to licensing rescritions) to `data_raw/articles`
3. Move files from `duc2004\results\ROUGE\eval\peers\2` to `data_raw/summaries`
4. Run `data_raw/import_corpus.py`
5. Copy `data_raw/corpus.pkl` to `cloned_summarizer/text_summarizer`
6. Models are avaliable in `src`. Example expirements avaliable in `Evaluate_DUC.ipynb`

## Centroid Embeddings:  

![](figs/Centroid_embedding.jpg) 

## Our Proposed Change: 

![](figs/LDA_centroid.jpg)

## Sentence embeddings:  

![](figs/sentence_representation.jpg)

## Centroid-sentence similarity: 

![](figs/centroid_sentence_similarity.jpg)

## Selection algorithm: 

![](figs/sentence_selection_algorithm.jpg)

## Rouge: 

![](figs/ROUGE.jpg)


