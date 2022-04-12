# %%
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import nltk
import torch
import pickle
from tqdm import tqdm
from scipy.spatial.distance import cosine
basecsvpath="/data/preped_annual_data/"
basepklspath="/data/pkl_sents/"
baseoutpath="/data/preped_annual_data/bert_embs/"

# %%
yearlist=["2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2019","2020"]

# keyword='performance'
keywordlist=['performance','attention','feature','programming','training','learning']


# %%
result={"word":[],"cos_dist_08_16":[]}
for keyword in keywordlist:
    year='2016'
    with open(baseoutpath+year+'_'+keyword+'_embs.pkl', 'rb') as f:
        prg_2020=pickle.load(f)
    year1='2008'
    with open(baseoutpath+year1+'_'+keyword+'_embs.pkl', 'rb') as f:
        prg_2008=pickle.load(f)
    cos_dist = 1 - cosine(prg_2020,prg_2008)
    result["word"].append(keyword)
    result["cos_dist_08_16"].append(cos_dist)

# %%
df_result=pd.DataFrame(result)
df_result.sort_values(by=["cos_dist_08_16"])
#save to local file
# srt_df.to_excel(basecsvpath+"_08_20.xls")


