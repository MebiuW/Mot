from Data.Model.Interfaces.Record.RawRecordAgent import *
#测试基本的QA
qas = [ RawQAPair('你好','什么'), RawQAPair('不错','好的')]
re = RawRecord(qas,'type',None)
save('record.txt',re)
rs = load('record.txt')
for item in rs.qas:
    print((item.question))
#测试从文件加载
rs = extract('TestCorpus.txt')
for qa in rs:
    print(repr(qa))