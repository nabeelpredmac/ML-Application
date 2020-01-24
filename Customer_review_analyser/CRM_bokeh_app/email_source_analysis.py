# Importing libraries
import imaplib, email
from tqdm import tqdm
import email_cred
import email, getpass, imaplib, os, re
import matplotlib.pyplot as plt
import html2text
import pandas as pd
  
user = email_cred.username
password = email_cred.password
imap_url = email_cred.imap_url
  
re_from = re.compile(u'From:')
re_date = re.compile(u'Date:')
re_subject = re.compile(u'Subject:')
re_slash = re.compile('\\r')

# Function to get email content part i.e its body part 
def get_body(msg): 
    if msg.is_multipart(): 
        return get_body(msg.get_payload(0)) 
    else: 
        return msg.get_payload(None, True) 


# Function to search for a key value pair  
def search(key, value, con):  
    result, data = con.search(None, key, '"{}"'.format(value))
    #result, data = con.search(None, key)
    return data 
  
# Function to get the list of emails under this label
def get_emails(result_bytes): 
    msgs = [] # all the email data are pushed inside an array 
    for num in result_bytes[0].split(): 
        typ, data = con.fetch(num, '(RFC822)') 
        msgs.append(data) 
  
    return msgs 

def main():
#if __name__=='__main__':
    # this is done to make SSL connnection with GMAIL 
    global con 
    con = imaplib.IMAP4_SSL(imap_url)  

    # logging the user in 
    con.login(user, password)  

    # calling fuction to check for email under this label 
    con.select('Inbox')  

    # fetching emails from this user "tu**h*****1@gmail.com" 
    # msgs = get_emails(search('FROM', 'roopeshpredmac@gmail.com', con)) 
    # msgs = get_emails(search('All', 'All', con)) 
    msgs = get_emails(search('Subject', 'review', con)) 

    froms = []
    dates = []
    subjects = []
    contents = []

    # printing them by the order they are displayed in your gmail  
    for msg in tqdm(msgs[::-1]):  
        for sent in msg: 
            if type(sent) is tuple:  

                # encoding set as utf-8 
                content = str(sent[1], 'utf-8')  
                data = str(content) 

                # Handling errors related to unicodenecode 
                try:  
                    indexstart = data.find("ltr") 
                    data2 = data[indexstart + 5: len(data)] 
                    ends = [m.start() for m in re.finditer('</div>', data2)]

                    span = re_from.search(data).span()[0]
                    span2 = data[span:-1].find('\r')
                    froms.append(data[span+6:span2+span])
                    span = re_date.search(data).span()[0]
                    span2 = data[span:-1].find('\r')
                    dates.append(data[span+6:span2+span])
                    span = re_subject.search(data).span()[0]
                    span2 = data[span:-1].find('\r')
                    subjects.append(data[span+9:span2+span])
                    contents.append(re.sub(' +', ' ',html2text.html2text(data2[0:ends[-1]]).replace('\n',' ')))
                    # printtng the required content which we need 
                    # to extract from our email i.e our body
                    df = pd.DataFrame()
                    df['date'] = dates
                    df['from'] = froms
                    df['title'] = subjects
                    df['comments'] = contents
                    print(re.sub(' +', ' ',html2text.html2text(data2[0:ends[-1]]).replace('\n',' '))) 

                except UnicodeEncodeError as e: 
                    pass

    df.to_csv('/home/gpu1/work_space/disk3_work_space3/CRM_Topic_models/input/review_from_mail.csv',index=False)
    
    return df