#/usr/bin/python

import string,os,sys
from Data.Model.Interfaces.BaseModelAgent import BaseModel
from Data.Model.SimpleImplement.HelloWorldModel import HWModel
mode = 'r'

model = HWModel()

if __name__ =='__main__':
    print('Hello ~ Welcome!')
    while(True):
        inputs = input('You:')
        if inputs == 'end':
            break
        print('Mbot:' + model.retrieve(inputs,1))

