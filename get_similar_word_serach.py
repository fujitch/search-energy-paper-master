# -*- coding: utf-8 -*-
"""
キーワードとその類似語を含む段落を検索する
検索結果:relative_sentences_dict
"""
import pickle

part_chapter_dict = pickle.load(open("data/figure_dict.pickle", "rb"))
model = pickle.load(open('model/word2vec_neologd100.pickle', 'rb'))


key_word = "石炭"
vector = model.wv[key_word]
out = model.most_similar([ vector ], [], 20)

relative_sentences_dict = {}

def not_include(obj_dict, obj_sentence):
    for k in obj_dict:
        objs = obj_dict[k]
        for obj in objs:
            if obj == obj_sentence:
                return False
    return True

for sim_word in out:
    if sim_word[1] < 0.6:
        break
    relative_sentences = []
    for k in part_chapter_dict:
        part_sentences = part_chapter_dict[k]
        for sentence in part_sentences:
            if sentence.find(sim_word[0]) != -1:
                if not_include(relative_sentences_dict, sentence):
                    relative_sentences.append(sentence)
    relative_sentences_dict[sim_word[0]] = relative_sentences