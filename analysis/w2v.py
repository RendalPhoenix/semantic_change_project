# %%
import re  
import pandas as pd  
from time import time  
from collections import defaultdict  
import numpy as np
from scipy.spatial.distance import cosine
basecsvpath="data/w2v_models"
from gensim.models import Word2Vec

# %%
years=['2008','2016','2020']
model_2008 = Word2Vec.load(basecsvpath+"_"+years[0]+"_word2vec_full.model")
model_2016 = Word2Vec.load(basecsvpath+"_"+years[1]+"_word2vec_full.model")
model_2020 = Word2Vec.load(basecsvpath+"_"+years[2]+"_word2vec_full.model")



# %% [markdown]
# #### Procrustes alignment for the models
# code from https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf

# %%
def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """
    Original script: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
    Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    # patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
    # base_embed.init_sims(replace=True)
    # other_embed.init_sims(replace=True)

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)
    # print(in_base_embed.wv.shape)
    # print(in_other_embed.wv.shape)
    # get the (normalized) embedding matrices

    in_base_embed.wv.fill_norms(force=True)
    in_other_embed.wv.fill_norms(force=True)

    base_vecs = in_base_embed.wv.get_normed_vectors()
    other_vecs = in_other_embed.wv.get_normed_vectors()

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v) 
    # Replace original array with modified one, i.e. multiplying the embedding matrix by "ortho"
    other_embed.wv.vectors = (other_embed.wv.vectors).dot(ortho)    
    
    return other_embed

def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.index_to_key)
    vocab_m2 = set(m2.wv.index_to_key)

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1,m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w, "count"), reverse=True)
    print(len(common_vocab))

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.key_to_index[w] for w in common_vocab]
        old_arr = m.wv.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        new_key_to_index = {}
        new_index_to_key = []
        for new_index, key in enumerate(common_vocab):
            new_key_to_index[key] = new_index
            new_index_to_key.append(key)
        m.wv.key_to_index = new_key_to_index
        m.wv.index_to_key = new_index_to_key
        
        print(len(m.wv.key_to_index), len(m.wv.vectors))
        
    return (m1,m2)

# %%
model_2020_aligned2_08=smart_procrustes_align_gensim(model_2008,model_2020)

# %%
## get most frequent 2000 words and calculate cosine dist across time
vb_2020=set(model_2008.wv.index_to_key[:2000])
vb_list=list(vb_2020)
vb_sim={"word":[],"cos_dist_08_20":[]}
for word in vb_list:
    vb_sim["word"].append(word)
    word_2020=model_2020_aligned2_08.wv[word]
    word_2008=model_2008.wv[word]    
    word_sim=1-cosine(word_2020,word_2008)
    vb_sim["cos_dist_08_20"].append(word_sim)

df_vb_sim=pd.DataFrame(vb_sim)
sorted_df=df_vb_sim.sort_values(by=['cos_dist_08_20'],ascending=False)
word_list=sorted_df['word'].to_list()
#save to local file
# sorted_df.to_excel(basecsvpath+"_most_feq_2000.xls")

# %%

similar_words_list_kw=[]
# keywords=['loss','train','solution','performance','attention','feature','training','learning','language']
keywords=['slack','loss','spark','train','training','react','language','performance','learning','attention','feature','table','value','file']

for kw in keywords:
    similar_words_list_year=[]
    kw_2020=model_2020_aligned2_08.wv[kw]
    kw_2008=model_2008.wv[kw]
    kw_similarity=1-cosine(kw_2020,kw_2008)
    similar_words_list_year.append(kw_similarity)

    similar_words_2008=model_2008.wv.similar_by_word(kw,topn=10)
    similar_words_string_2008=""
    for word in similar_words_2008:
        similar_words_string_2008+=word[0]+','
    similar_words_list_year.append(similar_words_string_2008)


    similar_words_2020=model_2020_aligned2_08.wv.similar_by_word(kw,topn=10)
    similar_words_string_2020=""
    for word in similar_words_2020:
        similar_words_string_2020+=word[0]+','
    similar_words_list_year.append(similar_words_string_2020)
    similar_words_list_kw.append(similar_words_list_year)

    
df = pd.DataFrame(similar_words_list_kw,
                index=pd.Index(keywords),
                columns=["cosine dist","2008","2020"])
srt_df=df.sort_values(by=['cosine dist'],ascending=False)
#save to local file
# srt_df.to_excel(basecsvpath+"_08_20.xls")


# %% [markdown]
# #### Plot for PCA visualization
# 
# the main func is a fork of the code here
# https://gist.github.com/marcellusruben/0be4e45eb342f664621166ed3c6e952f#file-pca_3d-py
# 
# modify it to visualize word vectors from different time periods in 2d
# 

# %%
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import plotly
import plotly.graph_objs as go
from sklearn.decomposition import PCA

# %%
def append_list(sim_words, words):
    
    list_of_words = []
    
    for i in range(len(sim_words)):
        
        sim_words_list = list(sim_words[i])
        sim_words_list.append(words)
        sim_words_tuple = tuple(sim_words_list)
        list_of_words.append(sim_words_tuple)
        
    return list_of_words



def pca_scatter(model_1,model_2, user_input=None, words=None, label=None, color_map=None, topn=10, sample=10):
    

    w1=words[0:10]
    w2=words[10:20]
    inputs=words[20:22]
    word_vectors_1 = np.array([model_1[w] for w in w1])
    word_vectors_2 = np.array([model_2[w] for w in w2])
    word_vectors_3 = np.array([model_1[w.split("-")[0]] for w in [inputs[0]]])
    word_vectors_4 = np.array([model_2[w.split("-")[0]] for w in [inputs[1]]])
    word_vectors=np.concatenate((word_vectors_1,word_vectors_2,word_vectors_3,word_vectors_4))
    two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]

    data = []
    count = 0
    
    for i in range (len(user_input)):

                trace = go.Scatter(
                    x = two_dim[count:count+topn,0], 
                    y = two_dim[count:count+topn,1],  
                    text = words[count:count+topn],
                    name = user_input[i],
                    textposition = "bottom center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 2
                    }
       
                )
                
            
                data.append(trace)
                count = count+topn

    trace_input = go.Scatter(
                    x = two_dim[count:,0], 
                    y = two_dim[count:,1],  
                    text = words[count:],
                    name = 'target word',
                    textposition = "bottom center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 1,
                        'color': 'black'
                    }
                    )
           
    data.append(trace_input)
    

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=0.4,
        y=-0.2,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1300
        )


    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.show()
    

# %%
ml_user_input = ['loss-2008','loss-2016']
ml_result_word = []

for words in ml_user_input:
    if words.split("-")[1]=="2008":
        sim_words = model_2008.wv.most_similar(words.split("-")[0], topn=10)
        sim_words = append_list(sim_words, words)
    else:
        sim_words = model_2020_aligned2_08.wv.most_similar(words.split("-")[0], topn=10)
        sim_words = append_list(sim_words, words)
        
    ml_result_word.extend(sim_words)

ml_similar_word = [word[0] for word in ml_result_word]
ml_similar_word.extend(ml_user_input)
labels = [word[2] for word in ml_result_word]
label_dict = dict([(y,x+1) for x,y in enumerate(set(labels))])
color_map = [label_dict[x] for x in labels]

# %%
pca_scatter(model_2008.wv,model_2020_aligned2_08.wv, ml_user_input, ml_similar_word, color_map)


