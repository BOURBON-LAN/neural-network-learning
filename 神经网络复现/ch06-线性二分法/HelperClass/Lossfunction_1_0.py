from itertools import count
from tkinter import N
import numpy as np
from EnumDef_1_0 import *

class Lossfuntion_1_0(object):
    def __init__(self,net_type):
        self.net_type = net_type

    def CE2(self,A,Y,count):
        #该函数计算的是二分类问题中的交叉熵损失。
        #A 是模型的预测值（通常是经过 sigmoid 激活函数后的概率值）。
        #Y 是真实标签（0 或 1）。
        #count 是样本数量，用于计算平均损失。
        #最终返回的是平均交叉熵损失。np.multiply是向量逐元素相乘
        p1 = 1-Y #计算真实标签Y的补集，即Y = 0时的概率
        p2 = np.log(1-A)#计算预测值A的补集的对数，即预测为Y=1的对数概率
        p3 = np.log(A)
        p4 = np.multiply(p1,p2)#计算 Y= 0时损失的部分，即真实标签为0的对数概率
        p5 = np.multiply(Y,p3)
        LOSS = np.sum(-(p4+p5))
        loss = LOSS/count
        return loss
    
    def CE2_tanh(self,A,Y,count):
        #若A是由于Tanh函数激活后的输出，范围在[-1,1]
        #计算损失部分的两个部分
        p = (1-Y)*np.log((1-A)/2) + (1+Y)*np.log((1+A)/2)
        #前部分为真实值标签y=-1时的损失部分，后者时Y=1时的损失部分
        LOSS = np.sum(-p)# 对损失求和，因为对数概率为负，损失应该为正
        loss = LOSS/count#count是样本数量
        return loss


    def Checkloss(self,A,Y):
        m = Y.shape[0]
        if self.net_type == NetType.BinaryClassifier:
            loss =  self.CE2(A,Y,m)
        elif self.net_type == NetType.BinaryTanh:
            loss = self.CE2_tanh(A,Y,count):
        return loss
    