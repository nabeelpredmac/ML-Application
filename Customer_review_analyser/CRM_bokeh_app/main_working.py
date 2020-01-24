# Pandas for data management
import pandas as pd

# os methods for manipulating paths
import os

# Bokeh basics 
from bokeh.io import show
from bokeh.io import show,curdoc
from bokeh.layouts import widgetbox,column,row
from bokeh.models.widgets import Dropdown,PreText, Select, Tabs

from bokeh.models import HoverTool,ColumnDataSource
from bokeh.plotting import curdoc,figure, output_file,save
from bokeh.models.widgets import Div

from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from datetime import datetime


# Each tab is drawn by one script
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

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.axis('off')
#     plt.margins(x=0, y=0)
#     plt.tight_layout()
#     plt.show()
    os.system('rm ./CRM_bokeh_app/static/images/*.png')
    now = str(datetime.now())
    plt.savefig('./CRM_bokeh_app/static/images/temp_'+now+'.png',bbox_inches='tight')
    
    footer_text = """
    <div >
    <img src="./CRM_bokeh_app/static/images/temp_"""+now+""".png" />
    </div>
    """
    div_footer = Div(text=footer_text,width=500,height=200)
    
    return div_footer
    
#def update(category,sentiment):
def update(attr, old, new):
    
    main_doc.clear()
    main_doc.add_root(layout)
    #main_doc.theme = 'dark_minimal'
    
    main_category = ticker0.value
    category = ticker1.value
    sentiment = ticker2.value
    reviews_ms1 = reviews.copy()
    
    if main_category!='all':
        reviews_ms1 = reviews[reviews['title']==main_category]
    
    if category!='all':
        reviews_ms1 = reviews_ms1[reviews_ms1['major_topic']==category]
    reviews_temp = reviews_ms1.copy()
    
    if sentiment != 'all':
        reviews_ms1 = reviews_ms1[reviews_ms1.sentiment_pred==sentiment]
        
    ## topics discussing by users about the main topic scrapping
    lda1,topics_lda1,topics_lda_df1,reviews_test_lda1,data_vectorized1,vectorizer1 = lda_topic_extraction.lda_finder(reviews_ms1,NUM_TOPICS=4)
    
    topic_distribution = pd.DataFrame(reviews_test_lda1.dominent_topic.value_counts()).reset_index()
    topic_distribution.columns = ['index','count']
    
    #source.data = source.from_df(topic_distribution[['index', 'count']])
    #push_notebook()
    if category !='all':
        reviews_test_lda1 = reviews_test_lda1.sort_values(by=category,ascending=False)
    reviews_test_lda1.to_csv('./CRM_bokeh_app/static/temp.csv',index=False)
    
    #for sentimental analysis distribution plot
    sentimental_distribution_df = pd.DataFrame(reviews_temp.sentiment_pred.value_counts()).reset_index()
    sentimental_distribution_df.columns = ['sentiment','count']
    
    z = plot_all(category,topic_distribution,topics_lda1,sentimental_distribution_df)
    z1 = plot_rows(reviews_ms1)
    z2 = column(column([z,ticker3],name='needed1'),column([z1],name='review1'),name='needed')
    
    main_doc.add_root(z2)
    #main_doc.add_root(z1)
    #main_doc.add_root(column([z1],name='review'))
    
def update2(attr, old, new):
    topic = ticker3.value
    temp = pd.read_csv('./CRM_bokeh_app/static/temp.csv')
    if topic!='all':
        temp = temp[temp.dominent_topic==int(topic)]
    
    z2 = column(plot_rows(temp) ,name='review1')
    #z2 = column([z2],name='review')
    rootLayout = main_doc.get_model_by_name('needed')
    listOfSubLayouts = rootLayout.children
    
    plotToRemove = main_doc.get_model_by_name('review1')
    listOfSubLayouts.remove(plotToRemove)
    
    listOfSubLayouts.append(z2)
    #main_doc.add_root(z2)

def plot_all(category,topic_distribution,topics_lda1,sentimental_distribution_df):

    p1 = figure(width=500, height=350, x_axis_type="linear",title="Topic distribution of "+category,name='review')#, y_range=[0, max_price+10])

    r_aapl = p1.vbar('index',.5 ,'count', source=topic_distribution, color='navy',  alpha=0.5)
    
    p2 = figure(width=500, height=350, title="Sentiment distribution of "+category,name='review',x_range=sentimental_distribution_df.sentiment)

    r_aapl = p2.vbar('sentiment',.5 ,'count', source=sentimental_distribution_df, color='navy',  alpha=0.5)
    
    p3 = word_plot1(topics_lda1)
    
    p0 = row([p1,p2])
    
    z = widgetbox([p0,p3])
    
    return z
    #main_doc.add_root(z)
    
    
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
    
    div_footer1 = Div(text=footer_text1,width=1300,height=200)
    
    
    return div_footer1
    #main_doc.add_root(column([div_footer1]))
      
        
## Reading the file
reviews = pd.read_excel('/home/gpu1/work_space/disk3_work_space3/CRM_Topic_models/data/Bag_Reviews.xlsx')
reviews = reviews[['rating','comments','title']]
reviews = reviews.drop_duplicates()
reviews.comments=reviews.comments.apply(lambda x: x.replace('👍','good '))

reviews['comments'] = reviews.comments.astype('str')
reviews['id'] = range(len(reviews))


## bokeh root init
main_doc = curdoc()

## reviews spliting as sentence and classifying them topic wise
reviews = data_clean_and_topic_detection.topic_finder(reviews)

## sentimental analysis
senti = sentimental_analysis.find_sentiment(reviews)
reviews['sentiment_pred'] = senti
## GUI part

# source = ColumnDataSource(data=dict(index=[], count=[]))
# p1 = figure(width=500, height=350, x_axis_type="linear",title="Topic distribution ")#, y_range=[0, max_price+10])

# r_aapl = p1.vbar('index',.5 ,'count', source=source, color='navy',  alpha=0.5)


menu = ['quality','price','beauty','delivery','all']
sentiment = ['all','negative','positive','neutral']
topics = ['all','0','1','2','3']
main_categories = ['all','Mini shoulder bag','shoulder bag','cute shoulder bag']

ticker0 = Select(title='Item', value='all', options=main_categories)
ticker0.on_change('value', update)

ticker1 = Select(title='category', value='quality', options=menu)
ticker1.on_change('value', update)

ticker2 = Select(title='sentiment', value='all', options=sentiment)
ticker2.on_change('value', update)

ticker3 = Select(title='topics', value='all', options=topics)
ticker3.on_change('value', update2)

controls = row([ticker0,ticker1, ticker2 ])#, width=200)
layout = row(controls)#, create_figure())

main_doc.add_root(layout)
main_doc.title = "CRM"
#main_doc.theme = 'dark_minimal'
update('','','')
