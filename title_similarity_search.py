# -*- coding: utf-8 -*-

"""
図表タイトルごとに類似の図表タイトルを検索する
"""

import pickle
import numpy as np
import MeCab
from gensim.models import Doc2Vec

# 1:word2vecモデル使用、2:doc2vecモデル使用
use_model_flg = 2

part_chapter_dict = pickle.load(open("data/energy_paper_2018_title.pickle", "rb"))

model = None
if use_model_flg == 1:
    model = pickle.load(open('model/word2vec_neologd100.pickle', 'rb'))
elif use_model_flg == 2:
    model = Doc2Vec.load('model/doc2vec_2_10_iter100.model')
    
m = MeCab.Tagger(r'-Owakati -d C:\Users\hori\workspace\encoder-decoder-sentence-chainer-master\mecab-ipadic-neologd')

# テキストのベクトルを計算word2vec
def get_vector(text):
    sum_vec = np.zeros(200)
    word_count = 0
    node = m.parseToNode(text)
    while node:
        fields = node.feature.split(",")
        # 名詞、動詞、形容詞に限定
        if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
            try:
                sum_vec += model.wv[node.surface]
            except:
                node = node.next
                continue
            word_count += 1
        node = node.next
    return sum_vec / word_count

# テキストのベクトルを計算doc2vec
def get_vector_doc(text):
    texts = m.parse(text).split(" ")
    return model.infer_vector(texts)

# cos類似度を計算
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# titleのリストに変換
title_list = []
for k in part_chapter_dict:
    for title in part_chapter_dict[k]:
        title_list.append(title)

similarity_dict = {}
similarity_sorted_title_dict = {}
for title in title_list:
    title_num = title[title.find("【"):title.find("】")+1]
    title_content = title[title.find("】")+1:]
    key_vector = None
    if use_model_flg == 1:
        key_vector = get_vector(title_content)
    elif use_model_flg == 2:
        key_vector = get_vector_doc(title_content)
    sim_list = []
    for t in title_list:
        t_content = t[t.find("】")+1:]
        t_vector = None
        if use_model_flg == 1:
            t_vector = get_vector(t_content)
        elif use_model_flg == 2:
            t_vector = get_vector_doc(t_content)
        if np.isnan(cos_sim(key_vector, t_vector)):
            sim_list.append(-np.inf)
        else:
            sim_list.append(cos_sim(key_vector, t_vector))
    similarity_dict[title] = sim_list
    
    sim_sorted_title_list = []
    index_list = np.argsort(sim_list)[::-1]
    for index in index_list:
        sim_sorted_title_list.append(title_list[index])
    similarity_sorted_title_dict[title] = sim_sorted_title_list

pickle.dump(similarity_dict, open("similarity_dict.pickle", "wb"))
pickle.dump(similarity_sorted_title_dict, open("similarity_sorted_title_dict.pickle", "wb"))