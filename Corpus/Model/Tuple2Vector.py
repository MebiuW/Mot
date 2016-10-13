# coding:utf-8
import sys
from gensim.models import Word2Vec

from numpy import *
import pynlpir

import logging,gensim,os
default_model = Word2Vec.load(r'C:\Users\72770\PycharmProjects\bot\qa_module\word2vector_6.model')
def tup2vec(tuples,model=default_model):
    #将QuestionToTuple中的对转化成实际的向量
    if len(tuples) == 0:
        return None
    tokens = [ word[0] for word in tuples if word[0] in model.vocab]
    weights = [ word[1] for word in tuples if word[0] in model.vocab]
    if len(tokens) == 0:
        return None
    init = model[tokens[0]] * weights[0]
    for i in range(1, len(tokens)):
        init = init + model[tokens[i]] * weights[i]
    return init