# Usual imports
import numpy as np
import pandas as pd
from tqdm import tqdm,tqdm_notebook
import string
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import concurrent.futures
import time
import pyLDAvis.sklearn
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig
import warnings
warnings.filterwarnings('ignore')

import os

# spaCy based imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import re

# Creating a spaCy object
nlp = spacy.load('en_core_web_lg')

punctuations = string.punctuation
stopwords = list(STOP_WORDS)

# Parser for reviews
parser = English()
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

# Functions for printing keywords for each topic
def selected_topics(model, vectorizer, top_n=10):
    topics = []
    for idx, topic in enumerate(model.components_):
        #print("Topic %d:" % (idx))
        topic_keys = [(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 0:-1]]
        topics.append((idx,topic_keys))
        #print([(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 0:-1]]) 
    return topics

##################################################################################################

def lda_finder(reviews_ms,NUM_TOPICS):
    
    # reviews_ms=reviews_ms.drop_duplicates()

    reviews_ms.comments=reviews_ms.comments.astype(str)
    reviews_ms['len_review']=reviews_ms.comments.apply(len)
    reviews_ms.comments=reviews_ms.comments.apply(lambda x: x.replace('👍','good '))

    s_limit=50
    max_limit=1300
    reviews=reviews_ms.loc[(reviews_ms.len_review>=s_limit) & (reviews_ms.len_review<max_limit),:]


    ## lemmetization, stopword remove, punctuation remove etc
    tqdm.pandas()
    reviews["processed_description"] = reviews["comments"].progress_apply(spacy_tokenizer)
    #reviews["processed_description"] = reviews["comments"].apply(spacy_tokenizer)

    # Creating a vectorizer
    vectorizer = CountVectorizer(min_df=0.005, max_df=0.85, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    data_vectorized = vectorizer.fit_transform(reviews["processed_description"])

    #NUM_TOPICS = 4
    
    SOME_FIXED_SEED = 46

    # before training/inference:
    np.random.seed(SOME_FIXED_SEED)
    
    # Latent Dirichlet Allocation Model
    #lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=50, learning_method='online'|batch,verbose=True)
    lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=50, 
                                    learning_method='online',verbose=False)#,random_state=1)
    data_lda = lda.fit_transform(data_vectorized)

    # Keywords for topics clustered by Latent Dirichlet Allocation
    #print("LDA Model:")
    topics_lda = selected_topics(lda, vectorizer)

    ## topics df with its words - distribution df
    topics_lda_df  = pd.DataFrame()
    i1 = [ t[0] for t in topics_lda]
    i2 = []
    for t in topics_lda:
        for t1 in t[1]:
            i2.append(t1[0])

    topics_lda_df['topic'] = i1
    for i in i2:
        topics_lda_df[i] = 0.0

    for i,t in enumerate(topics_lda):
        for t1 in t[1]:
            topics_lda_df.loc[topics_lda_df.topic==i,t1[0]]=t1[1]

    ## topic precentage in all reviews
    reviews_test_lda = reviews_ms.copy()#[(reviews_ms.len_review>=max_limit) | (reviews_ms.len_review<s_limit)]
    reviews_test_lda['index1'] = range(len(reviews_test_lda))

    dominent_topic_list = []
    topic_detail = pd.DataFrame()
    topic_detail['index1'] = reviews_test_lda.index1
    for i in range(0,NUM_TOPICS):
        topic_detail['topic_'+str(i)+'_perc'] = 0.0


    for i in tqdm(range(len(reviews_test_lda))):
        text = reviews_test_lda.comments.iloc[i]
        x = lda.transform(vectorizer.transform([text]))[0]
        y = pd.Series(x)
        for k in range(len(y)):
            topic_detail.loc[topic_detail.index1==i,'topic_'+str(k)+'_perc'] = y[k]
        y1 = y[y==max(y)].index[0]
        dominent_topic_list.append(y1)

    reviews_test_lda['dominent_topic'] = dominent_topic_list
    reviews_test_lda = reviews_test_lda.merge(topic_detail,on='index1',how='left')
    del reviews_test_lda['index1']

    return lda, topics_lda, topics_lda_df, reviews_test_lda, data_vectorized, vectorizer


