# coding:utf-8
import pynlpir
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

def sen2tuple(question):
    # 将问句从句子转化成<关键词,权重>的元组
    pynlpir.open(encoding='utf-8')
    try:
        keywords = pynlpir.get_key_words(question, weighted= True, max_words= 5)
        # 将结果改为Vector
        sum_of_weights = sum([ word[1] for word in keywords])
        vector = [(str(word[0]), word[1] / sum_of_weights )for word in keywords]
        return vector
    except Exception:
        return None


