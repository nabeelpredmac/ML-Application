import re 
from textblob import TextBlob 
from tqdm import tqdm
import pandas as pd

from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')


def clean_sentence(sentence): 
    ''' 
    Utility function to clean tweet text by removing links, special characters 
    using simple regex statements. 
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", sentence).split()) 

def get_sentence_sentiment(sentence): 
    ''' 
    Utility function to classify sentiment of passed tweet 
    using textblob's sentiment method 
    '''
    # create TextBlob object of passed tweet text 
    analysis = TextBlob(clean_sentence(sentence)) 
    # set sentiment 
    if analysis.sentiment.polarity > 0: 
        return 'positive'
    elif analysis.sentiment.polarity == 0: 
        return 'neutral'
    else: 
        return 'negative'

def find_sentiment(reviews):
    senti = []
    for i in tqdm(range(len(reviews))):
        senti.append(get_sentence_sentiment(reviews.comments.iloc[i]))

    return senti
    
def get_sentence_sentiment_stanford(sentence):
    sentence = clean_sentence(str(sentence))
    if len(sentence)>0:
        res = nlp.annotate(sentence,\
                       properties={\
                           'annotators': 'sentiment',\
                           'outputFormat': 'json',\
                       })
        senti = str(res["sentences"][0]["sentiment"].lower())
    else:
        senti = 'unknown'
    senti = senti.replace('very','')
    return senti

def find_sentiment_stanford(reviews):
    senti = []
    for i in tqdm(range(len(reviews))):
        senti.append(get_sentence_sentiment_stanford(reviews.comments.iloc[i]))

    return senti
