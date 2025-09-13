from turtle import forward
import numpy as np

class Logistic(object):
    def forward(self,z):
        #激活sigmoid函数的输出，将输出z映射到（0，1）之间
        a = 1.0/(1.0 + np.exp(-z))
        return a 
    
class Tahn(object):
    def forward(self,z):
        #激活Tahn函数的输出，即将输出z映射到（-1，1）之间
        a = 2.0 / (1.0 + np.exp(-2*z))-1.0
        return a 