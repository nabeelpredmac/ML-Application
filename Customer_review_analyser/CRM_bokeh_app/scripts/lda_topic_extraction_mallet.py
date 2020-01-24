import nltk
from nltk import FreqDist
nltk.download('stopwords') # run this one time

import pandas as pd
pd.set_option("display.max_colwidth", 200)
import numpy as np
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string

import gensim
from gensim import corpora

# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim
from tqdm import tqdm


punctuations = string.punctuation
stop_words = list(STOP_WORDS)

nlp = spacy.load('en', disable=['parser', 'ner'])

# function to remove stopwords
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new

def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output

def preprocess(text):
    text1 = remove_stopwords([text])

    # make entire text lowercase
    text1 = text1.lower()

    tokenized_text = text1.split()
    text2 = lemmatization([tokenized_text])
    return text2[0]

##################################################################################################

def lda_finder(reviews_ms,NUM_TOPICS):
    
    # reviews_ms=reviews_ms.drop_duplicates()

    reviews_ms.comments=reviews_ms.comments.astype(str)
    reviews_ms['len_review']=reviews_ms.comments.apply(len)
    reviews_ms.comments=reviews_ms.comments.apply(lambda x: x.replace('👍','good '))

    s_limit=50
    max_limit=1300
    reviews=reviews_ms.loc[(reviews_ms.len_review>=s_limit) & (reviews_ms.len_review<max_limit),:]

    # remove stopwords from the text
    reviews1 = [remove_stopwords(r.split()) for r in reviews['comments']]

    # make entire text lowercase
    reviews1 = [r.lower() for r in reviews1]
    
    tokenized_reviews = pd.Series(reviews1).apply(lambda x: x.split())
    reviews_2 = lemmatization(tokenized_reviews)
    
    dictionary = corpora.Dictionary(reviews_2)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]
    
    # Creating the object for LDA model using gensim library
    LDA = gensim.models.ldamodel.LdaModel

    # Build LDA model
#     lda = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=4, random_state=100,gamma_threshold=0.01,\
#                       minimum_probability=0.001,minimum_phi_value=0.001,chunksize=1000, passes=50,iterations=50,decay=.5,)
    
    mallet_path = '/home/gpu1/work_space/disk3_work_space3/CRM_Topic_models/mallet-2.0.8/bin/mallet' # update this path
    lda = gensim.models.wrappers.LdaMallet(mallet_path, corpus=doc_term_matrix, num_topics=4, \
                                           id2word=dictionary,iterations=50,random_seed=47)
    
    topics_lda = lda.show_topics(formatted=False)

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
        text = preprocess(text)
        x = lda[dictionary.doc2bow(text)]
        y = pd.Series([x1[1] for x1 in x])
        for k in range(len(y)):
            topic_detail.loc[topic_detail.index1==i,'topic_'+str(k)+'_perc'] = y[k]
        y1 = y[y==max(y)].index[0]
        dominent_topic_list.append(y1)

    reviews_test_lda['dominent_topic'] = dominent_topic_list
    reviews_test_lda = reviews_test_lda.merge(topic_detail,on='index1',how='left')
    del reviews_test_lda['index1']
    
    data_vectorized,vectorizer='',''

    return lda, topics_lda, topics_lda_df, reviews_test_lda, data_vectorized, vectorizer
