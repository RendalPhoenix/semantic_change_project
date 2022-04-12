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

yearlist=["2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2019","2020"]
keywordlist=['performance','attention','feature','programming','training','learning']


#hugging face implementation of BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True)


#collect sentences contain keywords
for year in tqdm(yearlist):
    filename="sentences_"+year+"_codermv.csv"
    df = pd.read_csv(basecsvpath+filename,nrows=300000)

    for keyword in keywordlist:
        contentarray=df.Content.tolist()

        word_found=0
        target_sent_list=[]
        for sent in contentarray: 
            if(str(sent).find(' '+keyword+' ')!=-1):
                word_found+=1
                target_sent_list.append(sent)

        for sent in target_sent_list:
            tokenized_text = tokenizer.tokenize(str(sent))
            if len(tokenized_text)>=400:
                target_sent_list.remove(sent)
        word_found2=0
        for sent in target_sent_list: 
            if(str(sent).find(keyword)!=-1):
                word_found2+=1
        with open(baseoutpath+year+'_'+keyword+'.pkl', 'wb') as f:
            pickle.dump(target_sent_list,f)


# get contextualized embeddings for keywords
for year in tqdm(yearlist):
    for keyword in keywordlist:
        with open(basecsvpath+year+'_'+keyword+'.pkl', 'rb') as f:
            sents_list=pickle.load(f)
        sents=sents_list
        sents.append(keyword)

        keyword_embeddings = []

        for sent in sents:
            tokenized_sents = tokenizer.tokenize("[CLS] " + sent + " [SEP]")
            tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_idx = [1]*len(tokens)

            tensor_tks = torch.tensor([tokens])
            tensors_segments = torch.tensor([segments_idx])

            if tensor_tks.shape[1]<512:
                with torch.no_grad():
                    hidden_states = model(tensor_tks, tensors_segments)[2][1:]   
                token_embeddings = hidden_states[-1]
                token_embeddings = torch.squeeze(token_embeddings, dim=0)
                for tk_embedding in token_embeddings:
                    token_embeddings_list=tk_embedding.tolist()
            else:
                continue

            # get the embedding for keyword
            word_embedding = token_embeddings_list[tokenized_sents.index(keyword)]

            keyword_embeddings.append(word_embedding)

        averaged_emd=np.sum(keyword_embeddings, axis=0)
        with open(baseoutpath+year+'_'+keyword+'_embs.pkl', 'wb') as f:
            pickle.dump(averaged_emd,f)
