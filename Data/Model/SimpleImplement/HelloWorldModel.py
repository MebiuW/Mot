from Data.Model.Interfaces.BaseModelAgent import BaseModel

class HWModel(BaseModel):
    '''
       简单的Hello World 模型,即输入什么,就输出什么
    '''

    def retrieve(self, question, k):
        return question


