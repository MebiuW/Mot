import pickle
class BaseModel:

    def train(self,qa_records):
        '''
        从0开始训练一个新的模型
        :param records:
        :return:
        '''
        pass

    def update(self,qa_records):
        '''
        加入新的预料进行更新
        :param qa_records:
        :return:
        '''

    def retrieve(self,question,k):
        '''
        根据Question,检索这个模型
        :param question:
        :return:
        '''
    def generate(selfs,question,k):
        '''
        根据Question,生成对应的回答
        :param question:
        :return:
        '''

    def save(file,model):
        '''
        将当前模型保存
        :param model:
        :return:
        '''
        output = open(file,'wb')
        pickle.dump(model,output)
        output.close()

    def load(file):
        '''
        加载当前模型
        :return:
        '''
        input  = open(file,'rb')
        model = pickle.load(input)
        input.close()
        return model