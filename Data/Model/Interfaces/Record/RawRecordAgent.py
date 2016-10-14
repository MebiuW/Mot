'''
一个RawRecord记录一个文件
'''
import pickle
import codecs

class RawQAPair():
    def __init__(self,question,answer,lang_type = 'unknown'):
        '''
        原始问题QA对
        :param question: 问题
        :param answer: 答案
        :param lang_type: 语言类型
        :return:
        '''
        self.question = question
        self.answer = answer
        self.lang_type = lang_type
        # 规范化,去除回车
        self.question.strip()
        self.answer.strip()
        self.question.replace('\n','<MBOT-SPACE>')

        def __str__(self):
            return '<' + self.question + ',' + self.answer + '>'

        #方便调试打印时候的显示
        def __repr__(self):
            return '<' + self.question + ',' + self.answer + '>'

class RawRecord():
    def __init__(self,qas,type,meta):
        '''

        :param qas:
        :param type:
        :return:
        '''
        self.qas = qas
        self.type = type
        self.meta = meta

    def getAllQaPairs(self):
        return self.qas

    def next(self):
        if self.qas is None:
            raise StopIteration
        for qa in self.qas:
            return  qa
        raise  StopIteration

    def __iter__(self):
        return self

    def getLangType(self):
        return self.type




def save(file,rawRecord):
    output = open(file,'wb')
    pickle.dump(rawRecord,output)
    output.close()

def load(file):
    input  = open(file,'rb')
    metadata = pickle.load(input)
    input.close()
    return metadata

def extract(file,type = 'unknown', encoding = 'utf-8'):
    '''
    从文件当中抽取出对应的数据
    :param file:
    :return:
    '''
    input  = codecs.open(file,mode='r',encoding=encoding)
    lines = input.readlines()
    state = -1
    rqas = list()
    # 按照行对文件进行解析
    ## 按照从开头开始
    for line in lines:
        line = line.strip()
        if line == '<Mbot-Context>':
            question = ''
            answer = ''
            state = 0
        elif line == '<Mbot-Answer>':
            state = 1
        elif line == '<Mbot-End>':
            if len(question) >0 and len(answer) > 0:
                rqas.append(RawQAPair(question,answer,type))
            state = -1
            question = ''
            answer = ''
        elif state == 0:
            question = question + line
        elif state == 1:
            answer = answer + line
    return rqas
