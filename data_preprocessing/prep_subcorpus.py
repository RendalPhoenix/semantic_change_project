#subsample the annual data and pre-process to 
# remove the html tag, URLs
# lower casing
# split into sentences
import numpy as np
import pandas as pd
import time
import tqdm
from bs4 import BeautifulSoup
import re
import os
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

basecsvpath="/data/annual_data/"
baseoutput="/data/sub_annual_data/"
baseoutput_sent="/data/preped_annual_data/"
yearlist=["2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"]


def cleanhtmltags(data):
    htmltext =BeautifulSoup(data, 'lxml').get_text()
    h5text = BeautifulSoup(htmltext, 'html5lib').get_text()
    text_cleanup = re.sub('<.*?>', ' ', str(h5text))  
    text_cleanup = re.sub('\\n', ' ', str(text_cleanup))  
    text_cleanup=re.sub('\sdiv\s', ' ', str(text_cleanup), flags=re.MULTILINE|re.DOTALL)
    text_cleanup=re.sub(r"http\S+", "", text_cleanup)
    return text_cleanup

# random subsample the annual data and save to local files 
for year in yearlist:
    # chunk read the big csv file to save memory 
    chunkreader = pd.read_csv(basecsvpath+"small_"+year+".csv", chunksize=100000) 
    count=0
    s_time_chunk = time.time()

    for chunk in chunkreader: 
        selected_df=chunk[['year','Id','CreationDate','Body','Title']]
        if year=="2008":
            selected_df = selected_df.sample(frac=0.7) # 70 percent 
        else:
            selected_df = selected_df.sample(frac=0.1) # 10 percent 
        count+=1
        if os.path.isfile(baseoutput_sent+'small_{}.csv'.format(year+"_sub")):  # check if file already exists
            selected_df.to_csv(baseoutput_sent+'small_{}.csv'.format(year+"_sub"), mode='a', index=False, header=False)
        else:
            selected_df.to_csv(baseoutput_sent+'small_{}.csv'.format(year+"_sub"), index=False)
        currentlines=count*100000
        if currentlines > 1 and (currentlines % 1000) == 0:
            print("{:,}".format(currentlines))
    e_time_chunk = time.time()

    print(year+"file done: ", (e_time_chunk-s_time_chunk), "sec")

#lower case the text and split the text into sentences
for year in yearlist:
    s_time_chunk = time.time()
    random_filename="small_"+year+"_sub.csv"
    random_df=pd.read_csv(basecsvpath+random_filename,nrows=140000)
    random_df.loc[random_df['Title'].isnull() , "Title"] = ""
    content_list=[]
    for row in tqdm(range(len(random_df))):
        title, body = random_df['Title'][row], random_df['Body'][row]
        removecode=re.sub('<code>(.*?)</code>', '', str(body))
        stripped_body=cleanhtmltags(str(removecode).encode('utf-8'))
        content=str(title)+" "+str(stripped_body)
        content_list.append(content)
    preprocessed = pd.DataFrame(zip(random_df['year'],random_df['Title'],content_list),columns= ["Year","Title","Content"])
    preprocessed['Content']=preprocessed['Content'].apply(lambda x: x.lower())
    preprocessed['Content']=preprocessed['Content'].apply(lambda x: sent_tokenize(x))
    preprocessed = preprocessed.explode("Content", ignore_index=True)
    sentenced_df=preprocessed[['Year','Content']]
    sentenced_df.to_csv(baseoutput_sent+"sentences_"+year+"_codermv.csv",index=False)
    
    e_time_chunk = time.time()
    print(year+"file done: ", (e_time_chunk-s_time_chunk), "sec")