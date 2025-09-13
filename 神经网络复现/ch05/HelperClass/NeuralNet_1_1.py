from functools import total_ordering
from matplotlib import axis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LogNorm

from HelperClass.DataReader_1_1 import*
from HelperClass.HyperParameters_1_0 import*
from HelperClass.TrainingHistory_1_0 import *

class NeuralNet_1_1(object):
    def __init__(self,hp):
        self.hp = hp
        self.W = np.zeros((self.hp.input_size,self.hp.output_size))
        self.B = np.zeros((1,self.hp.output_size))
    
    def __forwardBatch(self,batch_x):
        Z = np.dot(batch_x,self.W)+self.B
        return Z
    
    def __backwardBatch(self,batch_x,batch_y,batch_z):
        m = batch_x.shape[0]
        dZ = batch_z-batch_y
        dB = dZ.sum(0,True)/m
        dW = np.dot(batch_x.T,dZ)/m
        return dB,dW
    
    def __update(self,dW,dB):
        self.B = self.B - self.hp.eta*dB
        self.W = self.W - self.hp.eta*dW

    def inference(self,x):
        return self.__forwardBatch(x)
    
    def __checkLoss(self,dataReader):
        X,Y = dataReader.GetWholeTrainSamples()
        m = X.shape[0]
        Z = self.__forwardBatch(X)
        LOSS = (Z - Y)**2
        loss = LOSS.sum()/m/2
        return loss
    
    def train(self,dataReader,checkpoint=0.1):
        #计算损失函数并且决定终止条件,dataReader用来读取数据，checkpoint是用来表示按训练迭代次数的10%检查点的间隔
        loss_history = TrainingHistory_1_0()
        loss = 10
        if self.hp.batch_size ==-1:
            self.hp.batch_size = dataReader.num_train
        max_iteration = math.ceil(dataReader.num_train/self.hp.batch_size)
        checkpoint_iteration = (int)(max_iteration*checkpoint)

        for epoch in range (self.hp.max_epoch):
            print("epoch=%d" %epoch)
            dataReader.Shuffle()
            for iteration in range(max_iteration):
                batch_x,batch_y = dataReader.GetBatchTrainSample(self.hp.batch_size,iteration)
                batch_z = self.__forwardBatch(batch_x)
                dW, dB=self.__backwardBatch(batch_x,batch_y,batch_z)
                self.__update(dW,dB)

                total_iteration= epoch*max_iteration + iteration
                if (total_iteration+1)%checkpoint_iteration ==0:
                    loss = self.__checkLoss(dataReader)
                    print(epoch,iteration,loss,self.W,self.B)
                    loss_history.AddLossHistory(epoch*max_iteration,loss)
                    if loss < self.hp.eps:
                        break
        
        loss_history.ShowLossHistory(self.hp)
        print("W=",self.W)
        print("B=",self.B)

