#/usr/bin/python

import string,os,sys
from Data.Model.Interfaces.BaseModelAgent import BaseModel
from Data.Model.SimpleImplement.HelloWorldModel import HWModel
from Data.Model.SimpleImplement.SentenceSimilarityModel import SenSimModel
from Data.Model.SimpleImplement import  SentenceSimilarityModel
import datetime
mode = 'r'

model = SentenceSimilarityModel.load_model(r'C:/Users/72770/Documents/Chatbot/Model/SenHWModel/')

if __name__ =='__main__':
    print('Hello ~ Welcome!')
    while(True):
        inputs = raw_input('You:')
        if inputs == 'end':
            break
        start_time = datetime.datetime.now().microsecond
        print('Mbot:' + str(model.retrieve(inputs.decode('utf-8'),1).encode('utf-8')))

