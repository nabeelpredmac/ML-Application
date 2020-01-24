import nltk
import pandas as pd
import numpy as np
import re
import spacy
import gensim
import matplotlib.pyplot as plt
import seaborn as sns

from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm
import nltk.data
import string
from datetime import datetime
import sys

# os methods for manipulating paths
import os
from math import pi

# Bokeh basics 
from bokeh.io import show
from bokeh.io import show,curdoc
from bokeh.layouts import widgetbox,column,row
from bokeh.models.widgets import Dropdown,PreText, Select, Tabs,RadioGroup,RadioButtonGroup,TextInput
from bokeh.models import LabelSet, ColumnDataSource

from bokeh.models import HoverTool,ColumnDataSource
from bokeh.plotting import curdoc,figure, output_file,save
from bokeh.models.widgets import Div
from bokeh.palettes import Category20c
from bokeh.plotting import figure
from bokeh.transform import cumsum
from bokeh.core.properties import value

from gensim.summarization import summarize

import matplotlib.colors as mcolors

from scripts import sentimental_analysis

init_time = datetime.now()
nlp = spacy.load('en')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# Parser for reviews
parser = English()  


def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

def sentence_finder(reviews,limit1,limit2):
    
    full_comments = []
    full_title = []
    full_id = []
    for i in tqdm(range(limit1,limit2)):
        c = reviews.comments.iloc[i]
        c = tokenizer.tokenize(c)#c.split('.')
        full_comments = full_comments+c
        t1 = reviews.title.iloc[i]
        t = [t1 for j in range(len(c))]
        full_title = full_title+t
        id1 = reviews.id.iloc[i]
        id2 = [id1 for j in range(len(c))]
        full_id = full_id+id2
        
    return full_comments,full_title,full_id

def noun_finder(text):
    doc = nlp(text)
    all_nouns = []
    for i,token in enumerate(doc):
        if token.pos_ in ('NOUN','PROPN'):
            all_nouns.append(token)
    return all_nouns

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

threshold = 5
def junk_remover(a):
    a = a.split()
    max_val = max(pd.Series(a).value_counts())
    if max_val>threshold:
        return 1
    else:
        return 0
    
def update(attr, old, new):

    init_time = datetime.now()
    
    main_doc.clear()
    main_doc.add_root(layout)
    #main_doc.theme = 'dark_minimal'
    
    title = ticker0.value
    
    print('-I- selected item is : ',title)
    
#     sentiment = ticker1.value
    reviews = reviews_ms1.copy()
    reviews = reviews[reviews.title==title]
    reviews = reviews.drop_duplicates()
    reviews['id'] = range(len(reviews))

    ## sentence making
    full_comments = []
    full_title = []
    l1 = int(len(reviews)/2)
    c1,t1,id1 = sentence_finder(reviews,0,l1)
    c2,t2,id2 = sentence_finder(reviews,l1,len(reviews))
    full_comments = c1+c2
    full_title = t1+t2
    full_id    = id1+id2

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

    title_words = spacy_tokenizer(reviews.title.iloc[0].replace("'","")).split()
    def title_word_remover(s):
        for t in title_words:
            s = s.replace(t,'')
        return s

    ## For finding bigrams
    data = reviews.comments.values.tolist()
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]
    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=4, threshold=50) # higher threshold fewer phrases.
    # trigram = gensim.models.Phrases(bigram[data_words], threshold=50)  
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    # trigram_mod = gensim.models.phrases.Phraser(trigram)
    data_words1 = [' '.join(bigram_mod[d]) for d in data_words]
    reviews['comments'] = data_words1

    ## steming lemmetising and stop word removal
    reviews['cleaned_comments'] = reviews.comments.apply(spacy_tokenizer)

    ## sentiment finding
    senti = sentimental_analysis.find_sentiment(reviews)
    reviews['sentiment_pred'] = senti

    ## finding all nouns in the full reviews
    all_nouns = []
    for i in tqdm(range(len(reviews))):
        all_nouns = all_nouns+noun_finder(reviews.cleaned_comments.iloc[i])

    ## Nouns and their count with weight
    noun_df = pd.DataFrame(pd.Series(all_nouns).astype('str').value_counts().reset_index())
    noun_df.columns = ['noun','count']
    noun_df['weight'] = 0
    noun_df.head()

    print('-I- finding the weight and updating it in df ---------------------')
    
    ## finding the weight and updating it in df
    for text in tqdm(reviews.cleaned_comments):
        doc = nlp(text)
        noun_adj_pairs = []

        for i,token in enumerate(doc):
            bi_words = str(token).split('_')
            if ((token.pos_ not in ('ADJ')) & (len(bi_words)==1)):
                continue
            if ((len(bi_words)==2)):
                if((nlp(bi_words[0])[0].pos_=='ADJ')& (nlp(bi_words[1])[0].pos_ in ('NOUN','PROPN')) & (~pd.Series(bi_words[1]).isin(title_words)[0]) ):
                    noun_adj_pairs.append((bi_words[0],bi_words[1]))
                    try:
                        noun_df.loc[noun_df.noun==str(bi_words[1]),'weight'] = noun_df.loc[noun_df.noun==str(bi_words[1]),'weight'].iloc[0]+1
                    except:
                        noun_df = noun_df.append(pd.DataFrame({'noun':[bi_words[1]],'count':[1],'weight':[1]},index=[len(noun_df)]))
                elif((token.pos_ in ('NOUN','PROPN')) & (nlp(bi_words[0])[0].pos_ in ('NOUN','PROPN')) & (nlp(bi_words[1])[0].pos_ in ('NOUN','PROPN')) & (~pd.Series(bi_words[0]).isin(title_words)[0]) & (~pd.Series(bi_words[1]).isin(title_words)[0]) ):
    #             elif((nlp(bi_words[0])[0].pos_ in ('NOUN','PROPN')) & (nlp(bi_words[1])[0].pos_ in ('NOUN','PROPN'))):
                    noun_df.loc[noun_df.noun==str(token),'weight'] = noun_df.loc[noun_df.noun==str(token),'weight'].iloc[0]+1
                continue

            if((pd.Series([str(token)]).isin(positive)[0]) | (pd.Series([str(token)]).isin(negative)[0])):
                for j in range(i+1,min(i+6,len(doc))):
    #                 if (doc[j].pos_ in ('NOUN','PROPN')):
                    if ((doc[j].pos_ in ('NOUN','PROPN')) & (len(str(doc[j]).split('_'))!=2)):
                        noun_adj_pairs.append((token,doc[j]))
                        noun_df.loc[noun_df.noun==str(doc[j]),'weight'] = noun_df.loc[noun_df.noun==str(doc[j]),'weight'].iloc[0]+1
                        break

    ## removing words from noun which is in title to find top topics ( topic nouns ) 
    noun_df = noun_df[~noun_df.noun.isin(spacy_tokenizer(reviews.title.iloc[0].replace("'","")).split())]
    noun_df = noun_df.sort_values(by='weight',ascending=False)
    noun_df = noun_df.iloc[0:20,]
    reviews.to_csv('./CRM_bokeh_app/static/temp.csv',index=False)

    topic_df = pd.DataFrame()
    topic_df['topics']= noun_df['noun']
    print('-I- topic sentimental distribution finding---')
    pos_r = []
    neg_r = []
    neu_r = []
    full_l = []
    for t in tqdm(topic_df.topics):
        temp = reviews.copy()
        temp['item_presence'] = temp.apply(lambda row:topic_presence(row['cleaned_comments'],t),axis=1 )
        temp = temp[temp.item_presence!=-1]
        full_l.append(len(temp))
        pos_r.append(len(temp[temp.sentiment_pred=='positive']))
        neg_r.append(len(temp[temp.sentiment_pred=='negative']))
        neu_r.append(len(temp[temp.sentiment_pred=='neutral']))
#     topic_df['length'] = full_l
    topic_df['positive'] = pos_r
    topic_df['negative'] = neg_r
    topic_df['neutral'] = neu_r
    
    
    final_time = datetime.now()
    print('-I- Time taken for update is = ',str(final_time-init_time))

    global ticker2
    global radio_button_group
    global text_input
    
    ticker2 = Select(title='Topic', value='all', options=['all']+list(noun_df.noun))
    ticker2.on_change('value', update2)
    radio_button_group = RadioButtonGroup(labels=['all','positive','negative','neutral'], active=0)
    radio_button_group.on_change('active', update2)
    
    text_input = TextInput(value="", title="Custom topic search:")
    text_input.on_change("value", update2)

    
    t_r = column([row([ticker2,text_input]),radio_button_group])
    
    z = plot_senti(reviews,title)
    z1 = plot_topic(noun_df,title)
    z2 = column([row([plot_senti_stack(topic_df.sort_values(by='positive'),1),
                    plot_senti_stack(topic_df.sort_values(by='negative'),2)]),
                    plot_senti_stack(topic_df.sort_values(by='neutral'),3)])
    z = column([row([z,z1]),z2])
    
    z1 = row([plot_senti(reviews,'all topics'),summarizer(reviews)])
    z2 = plot_rows(reviews)
    z2 = column([z1,z2])
    z3 = column(column([z,t_r],name='needed1'),column([z2],name='review1'),name='needed')
    
    main_doc.add_root(z3)
    
def topic_presence(s,item):
    return str(s).find(item)

def plot_senti_stack(df,flag):
    topics = df.topics
    if flag==1:
        title = ' : Top positive'
        sentiment = ['positive','negative','neutral']
        colors = ["#c9d9d3", "#e84d60", "#718dbf"]
    elif flag==2:
        title = ' : Top negative'
        df = df[['topics','negative','positive','neutral']]
        sentiment = ['negative','positive','neutral']
        colors = ["#e84d60", "#c9d9d3", "#718dbf"]
    elif flag==3:
        title = ' : Top neutral'
        df = df[['topics','neutral','positive','negative']]
        sentiment = ['neutral','positive','negative']
        colors = ["#718dbf", "#c9d9d3", "#e84d60"]
    p = figure(y_range=topics, plot_height=350, title="Topic distribution with sentiment"+title,
           toolbar_location=None, tools="")

    p.hbar_stack(sentiment, y='topics', color=colors, source=df,height=0.9,# x_range=(-0.5, 1.0),
                 legend=[value(x) for x in sentiment])

    p.x_range.start = 0
    p.y_range.range_padding = 0.1
    p.ygrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None
    p.legend.location = "bottom_right"
    p.legend.orientation = "horizontal"
    
    return p

    
def update2(attr, old, new):
    
    text_topic = text_input.value
    topic = ticker2.value
    senti = radio_button_group.active
    senti_labels=['all','positive','negative','neutral']
    if len(text_topic)>0:
        topic = text_topic
    
    temp = pd.read_csv('./CRM_bokeh_app/static/temp.csv')
    if topic!='all':
        temp['item_presence'] = temp.apply(lambda row:topic_presence(row['cleaned_comments'],topic),axis=1 )
        temp = temp[temp.item_presence!=-1]
    
    print('-I- length of df = :',len(temp))
    z = plot_senti(temp,topic)
    
    if senti_labels[senti]!='all':
        temp = temp[temp.sentiment_pred==senti_labels[senti]]
    print(' Sentiment = ', senti_labels[senti])  
    z = row([z,summarizer(temp)])
    z2 = plot_rows(temp)
    z2 = column([z,z2] ,name='review1')
    #z2 = column([z2],name='review')
    rootLayout = main_doc.get_model_by_name('needed')
    listOfSubLayouts = rootLayout.children
    
    plotToRemove = main_doc.get_model_by_name('review1')
    listOfSubLayouts.remove(plotToRemove)
    
    listOfSubLayouts.append(z2)
    
def summarizer(df):
    df['comments'] = df.comments.astype('str')
    text = '. '.join(df.comments).replace('\n',' ')
    #text = reviews1[reviews.sentiment_pred=='positive'].comments.iloc[0]
    footer_text1 = """
    <div>
    <h1>------------------------ Reviews Summary ----------------------------------</h1></br>
    """
    if len(text)==0:
        footer_text1 = footer_text1+'No reviews present</br>'
    else:
        footer_text1 = footer_text1+str(summarize(text,ratio=.1,split=True,word_count=100))+'</br>'
        
    footer_text1 = footer_text1+'</div>'
    div_footer1 = Div(text=footer_text1,width=800,height=200)
    return div_footer1

def plot_senti(df,item):
    a = pd.DataFrame({'sentiment':['positive','negative','neutral'],'count':[0,0,0]})
    df1 = pd.DataFrame(df.sentiment_pred.value_counts()).reset_index()
    df1.columns = ['sentiment','count']
    df1 = a.merge(df1,on=['sentiment','count'],how ='outer').drop_duplicates(subset=['sentiment'],keep='last')
    
    ## pie chart
    df1['angle'] = df1['count']/df1['count'].sum() * 2*pi
    df1['color'] = Category20c[len(df1)]
    df1['sentiment'] = df1['sentiment'].astype('str')
    df1['perc'] = round((df1['count']/df1['count'].sum()),4)
    
    tooltips=[ ('Sentiment', '@sentiment'), ('Count','@count'),('Percentage ', '@perc{0.00%}')]
    p1 = figure(plot_height=350, toolbar_location=None, title="Sentiment distribution of '"+item+"'",\
               tools="hover", tooltips=tooltips, x_range=(-0.5, 1.0))

    p1.wedge(x=0, y=1, radius=0.3, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),\
            line_color="white", fill_color='color', legend='sentiment', source=df1)
    
    p1.axis.axis_label=None
    p1.axis.visible=False
    p1.grid.grid_line_color = None
    
    return p1

def plot_topic(df,item):
    df = df.sort_values(by='weight')
    p2 = figure(plot_width=500, plot_height=350, title="Topic weightage distribution of '"+item+"'",name='review',\
                y_range=df.noun, tooltips="Weight : @weight")

#     p2.hbar('weight',.5 ,'noun', source=df, color='navy',  alpha=0.5)
    p2.hbar(y='noun', height=0.5, left=0, right='weight',source=df, color="navy")
#     p2.yaxis.major_label_orientation = pi/2
    
    return p2

    
def plot_rows(reviews_ms1):
    print('--------------------------------------------------------------------------------')
    print('------------------------------ reviews -----------------------------------------')
    print('--------------------------------------------------------------------------------')
    
    footer_text1 = """
    <div>
    <h1>------------------------------ Reviews -----------------------------------------</h1></br>
    """
    if len(reviews_ms1)==0:
        footer_text1 = footer_text1+'No reviews present</br>'
    else:
        maxs=50
        if(len(reviews_ms1)<50):
            maxs = len(reviews_ms1)

        for i in range(maxs):
            full_s = str(reviews_ms1.comments.iloc[i])
            footer_text1 = footer_text1+full_s+" :-> "
            if(reviews_ms1.sentiment_pred.iloc[i]=='positive'):
                footer_text1 = footer_text1 + "<font color='green'>"+str(reviews_ms1.sentiment_pred.iloc[i])+"""</font></br>"""
            elif(reviews_ms1.sentiment_pred.iloc[i]=='negative'):
                footer_text1 = footer_text1 + "<font color='red'>"+str(reviews_ms1.sentiment_pred.iloc[i])+"""</font></br>"""
            elif(reviews_ms1.sentiment_pred.iloc[i]=='neutral'):
                footer_text1 = footer_text1 + "<font color='yellow'>"+str(reviews_ms1.sentiment_pred.iloc[i])+"""</font></br>"""
            
            footer_text1 = footer_text1+'-------------------------------------------------------------------------------- </br>'
    footer_text1 = footer_text1+'</div>'
    
    div_footer1 = Div(text=footer_text1,width=1300,height=200)
    return div_footer1
    #main_doc.add_root(column([div_footer1]))
        
######################### Main part ################################################

## reading the +ve and -ve sentiment words
positive = open('/home/gpu1/work_space/disk3_work_space3/CRM_Topic_models/input/positive-words.txt')
positive = positive.readlines()
positive = positive[31:]
print(len(positive))

negative = open('/home/gpu1/work_space/disk3_work_space3/CRM_Topic_models/input/negative-words.txt',encoding='utf8',errors='ignore')
negative = negative.readlines()
negative = negative[31:]

def replace_newline(text): return text.replace('\n','')

positive = list(pd.Series(positive).apply(replace_newline))
negative = list(pd.Series(negative).apply(replace_newline))

print(len(negative))

## panctuation and stopwords declaration
punctuations = string.punctuation
stopwords = list(STOP_WORDS)
stopwords = pd.Series(stopwords)
stopwords = list(stopwords[(~stopwords.isin(positive)) & (~stopwords.isin(negative))])

## Reading the reviews dataset
reviews_ms1 = pd.read_csv('/home/gpu1/work_space/disk3_work_space3/CRM_Topic_models/input/amazon_cloth_shoe_watches_reviews_with_title.csv')
reviews_ms1 = reviews_ms1[['rating','comments','title']]
reviews_ms1['comments'] = reviews_ms1.comments.astype('str')
reviews_ms1['title'] = reviews_ms1.title.astype('str')


## GUI part
## bokeh root init
main_doc = curdoc()

sentiment = ['all','negative','positive','neutral']
# main_categories = ['all','Mini shoulder bag','shoulder bag','cute shoulder bag']
main_categories = list(reviews_ms1.title.unique())

ticker0 = Select(title='Item', value=main_categories[0], options=main_categories)
ticker0.on_change('value', update)

# ticker1 = Select(title='sentiment', value='all', options=sentiment)
# ticker1.on_change('value', update)

controls = row([ticker0])#, width=200)
layout = row(controls)#, create_figure())

main_doc.add_root(layout)
main_doc.title = "CRM"
#main_doc.theme = 'dark_minimal'
update('','','')