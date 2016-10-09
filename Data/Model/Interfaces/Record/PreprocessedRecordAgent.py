from Data.Model.Interfaces.Record.RawRecordAgent import RawQAPair
from Data.Model.Interfaces.Record.RawRecordAgent import RawRecord
class QAPair(RawQAPair):
    '''
    被处理过的,主要是完成了分词和向量生成
    '''
    def __init__(self,question,answer,process_fun):
        super().__init__(question,answer)
        self.tokens = {'question':process_fun.tokenize(question),'answer':process_fun.tokenize(answer)}
        self.vector = {'question':process_fun.tovec(question),'answer':process_fun.tovec(answer)}

class QARecord(RawRecord):
    def __init__(self,qas,type,meta):
       super().__init__(qas,type,meta)