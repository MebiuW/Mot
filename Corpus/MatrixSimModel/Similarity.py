# coding:utf-8
import numpy as np
from mot.lib.ltp import  api as ltkapi
from gensim.models import Word2Vec
from gensim.models import Word2Vec
default_model = Word2Vec.load('word2vector_6.model')
cache = dict()

def weight_cossim(vector1,token1,weight1,vector2,token2):
    if token1 < token2:
        cache_line = token1+'LLL'+token2
    else:
        cache_line = token2+'LLL'+token1
    if cache_line in cache:
        return cache[cache_line]
    cosV12 = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    cache[cache_line] = cosV12
    return cosV12

def cossim(vector1,token1,vector2,token2):
    if token1 < token2:
        cache_line = token1+'LLL'+token2
    else:
        cache_line = token2+'LLL'+token1
    if cache_line in cache:
        return cache[cache_line]
    cosV12 = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    cache[cache_line] = cosV12
    return cosV12


def sen_sim(sentence1,sentence2,model=default_model):
    # 每个句子里的每个词，都寻找他在另外一个局子里醉相思的一个值
    # sentence1 = ltkapi.extract(sentence1)
    # sentence2 = ltkapi.extract(sentence2)

    tokens1 = ltkapi.segmentor(sentence1)
    tokens2 = ltkapi.segmentor(sentence2)
    tags1 = ltkapi.posttagger(tokens1)
    tags2 = ltkapi.posttagger(tokens2)
    arcs1 = ltkapi.parse(tokens1,tags1)
    arcs2 = ltkapi.parse(tokens2,tags2)
    weight = ltkapi.weight(arcs1)
    vec_of_tokens1 = [model[token] for token in tokens1 if token in model.vocab]
    vec_of_tokens2 = [model[token] for token in tokens2 if token in model.vocab]
    result = 0
    if len(vec_of_tokens1)==0 or len(vec_of_tokens2)==0:
        return result
    for i,(vector1,token1) in enumerate(zip(vec_of_tokens1,tokens1)):
        result = result + max([cossim(vector1,token1,vector2,token2) for vector2,token2 in zip(vec_of_tokens2,tokens2)])
    return result




