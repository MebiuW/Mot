# coding:utf-8
from Data.Model.Interfaces.BaseModelAgent import BaseModel
from Corpus.Model.Similarity import cossim
import gensim
from gensim.models.word2vec import LineSentence
from Data.Model.Interfaces.Record.RawRecordAgent import *
import jieba
import codecs
import json
import numpy as np



class SenSimModel(BaseModel):
    # 分词后的所有词典[词典->位置]
    tokens = dict()
    # 分词后所有词的词频
    tokens_fre = dict()
    #IDF
    idf = dict()
    # 所有词的数目
    word_count = 0

    def __init__(self,model_position):
        self.model_position = model_position

    def __DoStructureTrans(self,raw_qa_records):
        '''
        转化成一个以问题键值为key其他数据为value的dict
        :param raw_qa_records:
        :return:
        '''
        print ('======= Transform Your Corpus  =======')
        res = dict()
        for qa in raw_qa_records.getAllQaPairs():
            res[qa.question] = dict()
            res[qa.question]['context'] = qa.question
            res[qa.question]['response'] = qa.answer
        self.res = res

    def __DoBuildVocabulary(self):
        '''
        构建字典
        :param qa_records:
        :return:
        '''
        q_writer = codecs.open(self.model_position+r'/contexts-seg.txt','w+',encoding='utf-8')
        a_writer = codecs.open(self.model_position+r'/response-seg.txt','w+',encoding='utf-8')
        for qa in self.res:
            self.res[qa]['context_tokens'] = jieba.lcut(self.res[qa]['context'])
            self.res[qa]['response_tokens'] = jieba.lcut(self.res[qa]['response'])
            # 分词后将分词结果写入到txt中,进行Word2Vec的训练使用
            q_writer.write(' '.join(self.res[qa]['context_tokens']) + '\n')
            a_writer.write(' '.join(self.res[qa]['response_tokens']) + '\n')
            tokens = []
            tokens.extend(self.res[qa]['context_tokens'])
            tokens.extend(self.res[qa]['response_tokens'])
            for token in tokens:
                if token not in self.tokens:
                    self.tokens[token] = len(self.tokens)
                if token not in self.tokens_fre:
                    self.tokens_fre[token] = 0
                self.tokens_fre[token] = self.tokens_fre[token] + 1
            self.word_count = self.word_count + len(tokens)
            #IDF 值
            for token in set(tokens):
                # 设置好IDF
                if token not in self.idf.keys():
                    self.idf[token] = 0
                self.idf[token] = self.idf[token] + 1
        v_writer = open(self.model_position + r'/vocabulary.txt', 'w+')
        v_writer.write(json.dumps(self.tokens))
        v_writer = open(self.model_position + r'/vocabulary_fre.txt', 'w+')
        v_writer.write(json.dumps(self.tokens_fre))
        v_writer = open(self.model_position + r'/idf.txt', 'w+')
        v_writer.write(json.dumps(self.idf))


    def __DoWordEmbeddingTraining(self):
        '''
        使用word2vec训练
        :return:
        '''
        #新的语料库
        context_corpus = LineSentence(self.model_position+r'/contexts-seg.txt')
        response_corpus = LineSentence(self.model_position+r'/response-seg.txt')
        print ('======= Training Word2Vec Model =======')
        model = gensim.models.Word2Vec(context_corpus, workers=8)
        model.train(response_corpus)
        tmp_corpus = LineSentence(r'C:/Users/72770/Documents/Chatbot/Corpus/Noah Weibo/stc_weibo_train_post')
        tmp_corpus2 = LineSentence(r'C:\Users\72770\Documents\Chatbot\Corpus\Noah Weibo\stc_weibo_train_response')
        # model.train(tmp_corpus)
        # model.train(tmp_corpus2)
        print ('======= Saving Word2Vec Model =======')
        model.save(self.model_position+'/w2v.model')
        print ('======= Finished Word2Vec Traning =======')
        self.model = model


    def __DoVectorCreater(self):
        '''
        将已有的问句和回答都转变为Vector
        使用Hash压缩
        :return:
        '''
        print ('======= Training Model =======')
        m_writer = open(self.model_position + r'/model.obj', 'wb+')
        for qa in self.res:
            vector = []
            for token in self.res[qa]['context']:
                if token in self.model and self.idf[token] < 20:
                    vector.append(self.model[token])
            # 这里的Shape是n*100
            self.res[qa]['context_vector'] = np.array(vector)
            vector = []
            for token in self.res[qa]['response']:
                if token in self.model and self.idf[token] < 20:
                    vector.append(self.model[token])
            self.res[qa]['response_Vector'] = np.array(vector)
        print ('======= Saving Model =======')
        pickle.dump(self,m_writer)

    def train(self, qa_records):
        '''
        进行模型训练
        :param qa_records:
        :return:
        '''
        # 首先将其转换为可以识别的格式
        self.__DoStructureTrans(qa_records)
        # 然后分词 构建词典
        self.__DoBuildVocabulary()
        # 然后进行字典训练
        self.__DoWordEmbeddingTraining()
        # 然后进行向量生成训练
        self.__DoVectorCreater()
        #完成训练

    def __Sen2Vec(self,sentence):
        # 将句子的一段话转化成向量
        tokens = jieba.lcut(sentence)
        vecs = []
        for token in tokens:
            if token in self.model:
                vecs.append(self.model[token])
        return np.array(vecs)

    def __SenSim(self,vec_of_tokens1,vec_of_tokens2):
        result = 0
        if len(vec_of_tokens1) == 0 or len(vec_of_tokens2) == 0:
            return result
        for i, vector1 in enumerate(vec_of_tokens1):
            # TODO 使用矩阵改善这里
            result = result + max(
                [cossim(vector1, vector2) for vector2 in vec_of_tokens2]) / (0.0 + len(vec_of_tokens1))
        return result

    def retrieve(self,question,k):
        vector1 = self.__Sen2Vec(question)
        max_sim = -1
        max_answer = '这个问题我不太理解，换个问法试试？'
        for candidateQuestion in self.res:
            vector2 = self.res[candidateQuestion]['context_vector']
            cosV12 = self.__SenSim(vector1, vector2)
            if cosV12 > max_sim:
                max_sim = cosV12
                max_answer = self.res[candidateQuestion]['response']
        return max_answer

    def update(self,qa_records):
        pass


def load_model(modelPath):
    print ('======= Loading Model =======')
    obj =  pickle.load(open(modelPath+'model.obj'))
    print ('======= Model has been loaded =======')
    return obj

if __name__ == '__main__':
    print 'SENSIM---Model'
    input1 = codecs.open(r'C:/Users/72770/Documents/Chatbot/Corpus/Noah Weibo/stc_weibo_train_post', 'r',encoding='utf-8')
    input2 = codecs.open(r'C:/Users/72770/Documents/Chatbot/Corpus/Noah Weibo/stc_weibo_train_response', 'r',encoding='utf-8')
    qapairs = list()
    lx = len('Question '.decode('utf-8'))
    ly = len('Answer '.decode('utf-8'))
    for i in range(0,10000):
        qapairs.append(RawQAPair(input1.readline(),input2.readline()))
    record = RawRecord(qapairs,'Test',{})
    model = SenSimModel(r'C:/Users/72770/Documents/Chatbot/Model/SenHWModel/')
    model.train(record)
    # model = load_model(r'C:/Users/72770/Documents/Chatbot/Model/SenHWModel/')
    # print(model.retrieve('白条怎么用'.decode('utf-8'),8))
