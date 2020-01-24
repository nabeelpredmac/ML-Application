from numpy import dot
from numpy.linalg import norm
from itertools import chain

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import string
from collections import Counter

# spaCy based imports
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import WordNetLemmatizer 

punctuations = string.punctuation
stopwords = list(STOP_WORDS)
lemmatizer = WordNetLemmatizer()

# turn a doc into clean tokens
def clean_doc(doc):
    doc = doc.strip()
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuations)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    tokens = [w for w in tokens if not w in stopwords]
    # filter out short tokens
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if len(word) > 1]
    
    return tokens


## finding the major topic in each sentence
def max_topic_finder(quality1,delivery1,price1,beauty1):
    max_val = max(quality1,delivery1,price1,beauty1)
    if max_val == quality1:
        return 'quality'
    elif max_val == delivery1:
        return 'delivery'
    elif max_val == price1:
        return 'price'
    elif max_val == beauty1:
        return 'beauty'

threshold = 5
def junk_remover(a):
    a = a.split()
    max_val = max(pd.Series(a).value_counts())
    if max_val>threshold:
        return 1
    else:
        return 0
    
def sentence_finder(reviews,limit1,limit2):
    
    full_comments = []
    full_title = []
    full_id = []
    for i in tqdm(range(limit1,limit2)):
        c = reviews.comments.iloc[i]
        c = c.split('.')
        full_comments = full_comments+c
        t1 = reviews.title.iloc[i]
        t = [t1 for j in range(len(c))]
        full_title = full_title+t
        id1 = reviews.id.iloc[i]
        id2 = [id1 for j in range(len(c))]
        full_id = full_id+id2
        
    return full_comments,full_title,full_id
    
def topic_finder(reviews):

    ## sentence making
    full_comments = []
    full_title = []
#     for i in tqdm(range(len(reviews))):
#         c = reviews.comments.iloc[i]
#         c = c.split('.')
#         full_comments = full_comments+c
#         t1 = reviews.title.iloc[i]
#         t = [t1 for i in range(len(c))]
#         full_title = full_title+t

    l1 = int(len(reviews)/2)
    c1,t1,id1 = sentence_finder(reviews,0,l1)
    c2,t2,id2 = sentence_finder(reviews,l1,len(reviews))
    full_comments = c1+c2
    full_title = t1+t2
    full_id    = id1+id2
    
#     full_comments = list(reviews.comments)
#     full_title    = list(reviews.title)
    
    #print('length of comments',len(full_comments),'length of title =',len(full_title))

    #full_comments = pd.Series(full_comments).drop_duplicates()
    reviews_old = reviews[['id','comments']]
    reviews_old.columns = ['id','comments_full']
    
    def stripper(s): return s.strip()

    reviews = pd.DataFrame()
    reviews['comments'] = full_comments
    reviews['comments'] = reviews['comments'].apply(stripper)
    reviews['title'] = full_title
    reviews['id'] = full_id
    reviews = reviews[reviews.comments!='']
    reviews = reviews.drop_duplicates(subset=['comments'])
    reviews = reviews.merge(reviews_old,on='id',how='left')

    # removing junk reviews
    reviews['is_junk'] = 0
    reviews['is_junk'] = reviews.comments.apply(junk_remover)
    reviews = reviews[reviews.is_junk==0]
    del reviews['is_junk']
    
    # creating vocab
    vocab = sorted(set(token.lower() for token in chain(*list(map(clean_doc, reviews.comments)))))

    # main topics decleration
        #quality = [1 if token in ['quality','size','small','smaller','large','larger','stitching','service','suitable','good','bad','weak','strong','defective','comfortable','broken','damaged'] else 0 for token in vocab]
    quality = [1 if token in ['quality','stitching','service','suitable','weak','strong','defective','comfortable','broken','damaged'] else 0 for token in vocab]
    delivery = [1 if token in ['delivery','deliver','delivered','fast','slow','slowly','late','shipping','shipped','send','received','receive','ship','shop','arrived','order','ordered','store'] else 0 for token in vocab]
    price = [1 if token in ['price','cheap','money','discount','offer','value','worth','purchase','refund','expensive','worthwhile','affordable'] else 0 for token in vocab]
    beauty = [1 if token in ['beautiful','color','beauty','cute','shape','style','luxury','luxurious','looks','attractive','unattractive','impressive','picture'] else 0 for token in vocab]

    topics = {'quality': quality, 'delivery': delivery, 'price': price, 'beauty':beauty}

    reviews['tokens'] = reviews.comments.apply(clean_doc)
    reviews = reviews[reviews.tokens.apply(len)!=0]
    reviews.index = range(len(reviews))

    ## vectorizing
    def vectorize(l):
        return [1 if token in l else 0  for token in vocab]
    
    reviews['vectors'] = reviews.tokens.apply(vectorize)

    ## Finding similirity of each sentence with each topic
    reviews['quality']  = 0.0
    reviews['delivery'] = 0.0
    reviews['price']    = 0.0
    reviews['beauty']   = 0.0

    cos_sim = lambda x, y: dot(x,y)/(norm(x)*norm(y))

    for s_num, s_vec in enumerate(reviews.vectors):
        for name, topic_vec in topics.items():
            similarity = cos_sim(s_vec, topic_vec)
            reviews.loc[s_num,name] = similarity


    reviews['major_topic'] = reviews.apply(lambda row:max_topic_finder(row['quality'],row['delivery'],row['price'],row['beauty']),axis=1)

    return reviews