# split the csv data by year
import numpy as np
import pandas as pd
import time
import os
csvpath="/data/Posts.csv"
basepath="/data/annual_data/"

s_time_chunk = time.time()
chunk = pd.read_csv(csvpath, chunksize=100)
count=0
yearlist=["2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"]
for ck in chunk:
    count+=1
    ck['year']=ck['CreationDate'].str[0:4]
    for year in yearlist:
        df_to_write = ck[ck['year'] == year]

        if os.path.isfile(basepath+'small_{}.csv'.format(year)):  # check if file already exists
            df_to_write.to_csv(basepath+'small_{}.csv'.format(year), mode='a', index=False, header=False)
        else:
            df_to_write.to_csv(basepath+'small_{}.csv'.format(year), index=False)
    currentlines=count*100
    if currentlines > 1 and (currentlines % 1000) == 0:
        print("{:,}".format(currentlines))

e_time_chunk = time.time()

print("all files done: ", (e_time_chunk-s_time_chunk), "sec")
