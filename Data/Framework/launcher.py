#/usr/bin/python

import string,os,sys
from Data.Model.Interfaces.BaseModelAgent import BaseModel
from Data.Model.SimpleImplement.HelloWorldModel import HWModel
from Data.Model.SimpleImplement.SentenceSimilarityModel import SenSimModel
import datetime
mode = 'r'

model = SenSimModel('')

if __name__ =='__main__':
    print('Hello ~ Welcome!')
    while(True):
        inputs = raw_input('You:')
        if inputs == 'end':
            break
        start_time = datetime.datetime.now().microsecond
        print('Mbot:' + model.retrieve(inputs,1))

