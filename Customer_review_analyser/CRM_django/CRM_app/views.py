# from django.shortcuts import render, render_to_response
# from django.http import HttpResponse

# # Create your views here.
# def home(request):
#     return HttpResponse('<h1> This is the Home page </h1>')

#<select name = "category1" onchange="this.form.submit()" >
# <script type="text/javascript">
#     document.getElementById('category1').value = {{ category_s }};
# </script>

from django.shortcuts import render
from django import forms

from bokeh.plotting import figure, output_file,save
from bokeh.embed import components,file_html
from bokeh.models import HoverTool, LassoSelectTool, WheelZoomTool, PointDrawTool, ColumnDataSource
from django.views.decorators.csrf import csrf_protect
from django.template import RequestContext
from django.contrib.auth.decorators import login_required

from bokeh.palettes import Category20c, Spectral6
from bokeh.transform import cumsum
from bokeh.models.widgets import Dropdown,PreText, Select, Tabs
from bokeh.layouts import widgetbox,column,row
from bokeh.models.widgets import Div
from numpy import pi
import pandas as pd
from bokeh.resources import CDN

from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from datetime import datetime
import os

# Each tab is drawn by one script
import sys
sys.path.insert(0,'./CRM_app')
from scripts import data_clean_and_topic_detection
from scripts import lda_topic_extraction
from scripts import sentimental_analysis

def word_plot1(topics):
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    #topics = lda.show_topics(formatted=False)
    # 14,14
    fig, axes = plt.subplots(1, 4, figsize=(14,14), sharex=True, sharey=True)

    #<form action={% url 'CRM-home' %} method='POST' name='myform'>
    
    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    print('current dir =',os.getcwd())
    os.system('rm ./CRM_app/static/images/*.png')
    now = str(datetime.now()).replace(' ','_')
    plt.savefig('./CRM_app/static/images/temp_'+now+'.png',bbox_inches='tight')
    return './../static/images/temp_'+now+'.png'

def plot_all(category,topic_distribution):

    p1 = figure(width=400, height=250, x_axis_type="linear",title="Topic distribution of "+category,name='review')#, y_range=[0, max_price+10])

    r_aapl = p1.vbar('index',.4 ,'count', source=topic_distribution, color='navy',  alpha=0.4)
        
    return p1

def plot_rows(reviews_ms1):
    print('--------------------------------------------------------------------------------')
    print('------------------------------ reviews -----------------------------------------')
    print('--------------------------------------------------------------------------------')
    
    footer_text1 = """
    <div>
    <h1>------------------------------ Reviews -----------------------------------------</h1></br>
    """
    maxs=50
    if(len(reviews_ms1)<50):
        maxs = len(reviews_ms1)
        
    for i in range(maxs):
        l1 = reviews_ms1.comments_full.iloc[i].find(reviews_ms1.comments.iloc[i])
        l2 = len(reviews_ms1.comments.iloc[i]) + l1
        l3 = len(reviews_ms1.comments_full.iloc[i])
        full_s = reviews_ms1.comments_full.iloc[i]
        footer_text1 = footer_text1+full_s[0:l1]+"""<font color='green'>"""+full_s[l1:l2]+"""</font>"""+full_s[l2:l3]+"""</br>"""
        footer_text1 = footer_text1+'-------------------------------------------------------------------------------- </br>'
    footer_text1 = footer_text1+'</div>'
    
    #div_footer1 = Div(text=footer_text1,width=1300,height=200)
        
    return footer_text1#div_footer1


def update(reviews, item_s, category_s, sentiment_s):
    
    reviews_ms1 = reviews.copy()
    
    if item_s!='all':
        reviews_ms1 = reviews_ms1[reviews_ms1['title']==item_s]
    
    if category_s!='all':
        reviews_ms1 = reviews_ms1[reviews_ms1['major_topic']==category_s]
        
    if sentiment_s != 'all':
        reviews_ms1 = reviews_ms1[reviews_ms1.sentiment_pred==sentiment_s]
    print('length of the combination df = ',len(reviews_ms1))
    
    if len(reviews_ms1)>50:    
        ## topics discussing by users about the main topic scrapping
        lda1,topics_lda1,topics_lda_df1,reviews_test_lda1,data_vectorized1,vectorizer1 = lda_topic_extraction.lda_finder(reviews_ms1,NUM_TOPICS=4)

        topic_distribution = pd.DataFrame(reviews_test_lda1.dominent_topic.value_counts()).reset_index()
        topic_distribution.columns = ['index','count']

        #source.data = source.from_df(topic_distribution[['index', 'count']])
        #push_notebook()
        if category_s !='all':
            reviews_test_lda1 = reviews_test_lda1.sort_values(by=category_s,ascending=False)
        reviews_test_lda1.to_csv('./CRM_app/static/temp.csv',index=False)

        p1 = plot_all(category_s,topic_distribution)
        now = word_plot1(topics_lda1) 
        table_path = './CRM_app/static/temp.csv'
    
    else:
        topic_distribution = pd.DataFrame({'index':[],'count':[]})
        p1 = plot_all(category_s,topic_distribution)
        now = './static/images/insufficient_data.jpg'
        reviews_test_lda1 = pd.DataFrame({'comments':[],'comments_full':[],'id':[]})
        reviews_test_lda1.to_csv('./CRM_app/static/temp.csv',index=False)
        table_path = './CRM_app/static/temp.csv'
        
    return p1,now,table_path

def update2(table_path, topic_s):
    temp = pd.read_csv(table_path)
    if topic_s!='all':
        temp = temp[temp.dominent_topic==int(topic_s)]
        
    html_code = plot_rows(temp)
        
    return html_code

t1 = datetime.now()
## Reading the file
reviews = pd.read_excel('/home/gpu1/work_space/disk3_work_space3/CRM_Topic_models/data/Bag_Reviews.xlsx')
reviews = reviews[['rating','comments','title']]
reviews = reviews.drop_duplicates()
reviews.comments=reviews.comments.apply(lambda x: x.replace('👍','good '))
reviews['comments'] = reviews.comments.astype('str')
reviews['id'] = range(len(reviews))
t2 = datetime.now()
print('-I- time taken to read data : ',str(t2-t1))

## reviews spliting as sentence and classifying them topic wise
reviews = data_clean_and_topic_detection.topic_finder(reviews)
t3 = datetime.now()
print('-I- time taken to clean_and_topic_detection of data : ',str(t3-t2))

## sentimental analysis
senti = sentimental_analysis.find_sentiment(reviews)
reviews['sentiment_pred'] = senti

t4 = datetime.now()
print('-I- time taken to sentimental anlysis of data : ',str(t4-t3))


@login_required
@csrf_protect
def home(request):
    
    t4 = datetime.now()
    
    print('Request from georgee is ',request.method)
    print('values : ',request.POST.values)
    
    csrfContext = RequestContext(request)

    item_s = 'all'
    category_s='quality'
    sentiment_s='all'
    topic_s = 'all'
    if request.POST:
        print('the request is :',request.POST['category1'])
        item_s = request.POST['item']
        category_s = request.POST['category1']
        sentiment_s = request.POST['sentiment']
        topic_s = request.POST['topics']
    
    item = ['all','Mini shoulder bag','shoulder bag','cute shoulder bag']
    category = ['quality','price','beauty','delivery','all']
    sentiment = ['all','negative','positive','neutral']
    topics = ['all','0','1','2','3']

    p1,now,table_path = update(reviews, item_s, category_s, sentiment_s)
    html_code = update2(table_path,topic_s)
    
    t5 = datetime.now()
    print('-I- time taken for update operation : ',str(t5-t4))
    
    script, div = components(p1)
    
    image_loc = {'image': now,'script': script, 'div':div,'item':item,'category':category,'sentiment':sentiment,
                'topics':topics,'item_s':item_s,'category_s':category_s,'sentiment_s':sentiment_s,
                 'topic_s':topic_s,'html_code':html_code}
    
    
    return render(request, 'home.html', image_loc)


from django.contrib.auth import authenticate, login

