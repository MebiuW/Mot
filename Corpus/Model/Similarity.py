# coding:utf-8
import numpy as np

def cossim(vector1,vector2):
    cosV12 = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return cosV12
