import re  
import pandas as pd
from time import time  
from collections import defaultdict  
from tqdm import tqdm
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
import spacy  
import multiprocessing


basecsvpath="data/preped_annual_data/"
yearlist=["2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"]
for year in yearlist:
    filename_16="sentences_"+year+"_codermv.csv"
    df=pd.read_csv(basecsvpath+filename_16)

    en_spy = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

    def rmv_stpwds(sents):
        text = [token.lemma_ for token in sents if not token.is_stop]
        if len(text) > 2:
            return ' '.join(text)

    clean_symbls = (re.sub("[^A-Za-z']+", ' ', str(row)) for row in df['Content'])
    cleanedup_sents = [rmv_stpwds(sents) for sents in tqdm(en_spy.pipe(clean_symbls, batch_size=1000))]


    df_cleanedup = pd.DataFrame({'clean': cleanedup_sents}).dropna().drop_duplicates()
    sent = [row.split() for row in df_cleanedup['clean']]
    bigram = Phraser(Phrases(sent, min_count=30, progress_per=10000))
    sentences = bigram[sent]

    #setting up word2vec model params, d=300
    w2v_model = Word2Vec(min_count=20,
                        window=2,
                        vector_size=300,
                        sample=6e-5, 
                        alpha=0.03, 
                        min_alpha=0.0007, 
                        negative=20,
                        workers=multiprocessing.cpu_count()-2)
    w2v_model.build_vocab(sentences, progress_per=10000)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    w2v_model.init_sims(replace=True)
    w2v_model.save(basecsvpath+"_"+year+"_word2vec_full.model")