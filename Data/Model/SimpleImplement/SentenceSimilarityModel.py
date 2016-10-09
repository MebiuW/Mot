from Data.Model.Interfaces.BaseModelAgent import BaseModel
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
        writer = open(self.model_position,'w+')
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
        pass

    def __DoVectorCreater(self):
        pass



    def train(self, qa_records):
        '''
        进行模型训练
        :param qa_records:
        :return:
        '''
        super().train(qa_records)
        # 首先将其转换为可以识别的格式
        self.__DoStructureTrans(qa_records)
        # 然后分词 构建词典
        self.__DoVectorCreater()
        # 然后进行字典训练
        self.__DoWordEmbeddingTraining()
        # 然后进行向量生成训练
        self.__DoVectorCreater()
        #完成训练