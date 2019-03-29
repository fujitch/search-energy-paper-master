# -*- coding: utf-8 -*-

"""
キーワード、キーセンテンスから類似度の高い段落を抽出する
抽出結果:simirality_sentences_sort_dict
"""
import pickle
import numpy as np
import MeCab
from gensim.models import Doc2Vec

# 1:段落内の類似度最大の文から抽出、2:各文の類似度の平均値から段落を計算(word2vec使用時のみ)
max_mean_flg = 2

# 1:word2vecモデル使用、2:doc2vecモデル使用
use_model_flg = 2

# キーワード、キーセンテンス
key_sentence = "なお、二次エネルギーである電気は家庭用及び業務用を中心にその需要は2000年代後半まで増加の一途をたどりました。電力化率3は、1970年度には12.7%でしたが、2016年度には25.7%に達しました（第211-3-3）"

part_chapter_dict = pickle.load(open("data/figure_dict.pickle", "rb"))

# model初期化
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

key_vector = None
if use_model_flg == 1:
    key_vector = get_vector(key_sentence)
elif use_model_flg == 2:
    key_vector = get_vector_doc(key_sentence)

simirality_sentences_dict = {}
for k in part_chapter_dict:
    print(k)
    part_sentences = part_chapter_dict[k]
    for paragraph in part_sentences:
        all_score = 0.0
        if use_model_flg == 1:
            counter = 0
            texts = paragraph.split("。")
            if texts[len(texts)-1] == "":
                texts = texts[:len(texts)-1]
            for text in texts:
                vec = get_vector(text)
                score = cos_sim(key_vector, vec)
                if max_mean_flg == 1:
                    if all_score < score:
                        all_score = score
                elif max_mean_flg == 2:
                    all_score += score
                counter += 1
            if max_mean_flg == 2:
                all_score /= counter
        elif use_model_flg == 2:
            vec = get_vector_doc(paragraph)
            all_score = cos_sim(key_vector, vec)
        simirality_sentences_dict[all_score] = paragraph

simirality_sentences_sort_dict = {}
for k, v in sorted(simirality_sentences_dict.items(), key=lambda x: -x[0]):
    simirality_sentences_sort_dict[k] = v
