# coding:utf-8
from SentenceToTuple import sen2tuple
from Tuple2Vector import tup2vec

def sen2vec(sentence):
    #将句子的一段话转化成向量
    tuple = sen2tuple(sentence)
    if tuple is None:
        return None
    vec = tup2vec(tuple)
    if vec is None:
        return None
    return vec