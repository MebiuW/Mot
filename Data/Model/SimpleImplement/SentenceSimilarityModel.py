# coding:utf-8
from Data.Model.Interfaces.BaseModelAgent import BaseModel
from Corpus.Model.Sentence2Vector import sen2vec
from Corpus.Model.Similarity import cossim
import gensim
from gensim.models.word2vec import LineSentence
import jieba


class SenSimModel(BaseModel):
    # 分词后的所有词典[词典->位置]
    tokens = dict()
    # 分词后所有词的词频
    tokens_fre = dict()
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
        writer = open(self.model_position+r'/seg.txt','a+')
        for qa in self.res:
            self.res[qa]['context_tokens'] = jieba.lcut(self.res[qa]['context'])
            self.res[qa]['response_tokens'] = jieba.lcut(self.res[qa]['response'])
            # 分词后将分词结果写入到txt中,进行Word2Vec的训练使用
            writer.write(' '.join(self.res[qa]['context_tokens']) + '\n')
            writer.write(' '.join(self.res[qa]['response_tokens']) + '\n')
            tokens = []
            tokens.extend(self.res[qa]['context_tokens'])
            tokens.extend(self.res[qa]['response_tokens'])
            for token in tokens:
                if token not in self.tokens:
                    self.tokens[token] = len(self.tokens)
                if token not in self.tokens_fre:
                    self.tokens_fre[token] = 0
                self.tokens_fre[token] = self.tokens_fre[token] + 1
            self.word_count = self.word_count + len(self.word_count)


    def __DoWordEmbeddingTraining(self):
        '''
        使用word2vec训练
        :return:
        '''
        #新的语料库
        new_corpus = LineSentence('corpus/iphone6sreview-seg.txt')
        print ('======= Training Word2Vec Model =======')
        model = gensim.models.Word2Vec(new_corpus, workers=8)
        print ('======= Saving Word2Vec Model =======')
        model.save(self.model_position+'/w2v.model')
        print ('======= Finished Word2Vec Traning =======')
        self.model = model


    def __DoVectorCreater(self):
        '''
        将已有的问句和回答都转变为Vector
        :return:
        '''
        for qa in self.res:
            self.res[qa.question]['question_vector'] = [0,0]
            self.res[qa.question]['answer_vector'] = [0,0]
        pass



    def train(self, qa_records):
        '''
        进行模型训练
        :param qa_records:
        :return:
        '''
        super.train(qa_records)
        # 首先将其转换为可以识别的格式
        self.__DoStructureTrans(qa_records)
        # 然后分词 构建词典
        self.__DoVectorCreater()
        # 然后进行字典训练
        self.__DoWordEmbeddingTraining()
        # 然后进行向量生成训练
        self.__DoVectorCreater()
        #完成训练

    def retrieve(self,question,k):
        vector1 = sen2vec(question)
        input = open(r'C:\Users\72770\PycharmProjects\Crawler\JDQATemp\iphone6s-record-qa.txt', 'r')
        # 问题
        line = input.readline()
        max_sim = -1
        max_answer = '这个问题我不太理解，换个问法试试？'
        while line != 'EOF' and len(line) > 0:
            try:
                # 答案对
                answer = input.readline()
                if len(answer) < 10:
                    continue
                line = line[8:]
                vector2 = sen2vec(str(line))
                if vector2 is None:
                    continue
                cosV12 = cossim(vector1, vector2)
                if cosV12 > max_sim:
                    max_sim = cosV12
                    max_answer = answer
            except Exception as e:
                pass
                print('Some Error' + str(e))
            finally:
                line = input.readline()
        return str(max_answer)

    def update(self,qa_records):
