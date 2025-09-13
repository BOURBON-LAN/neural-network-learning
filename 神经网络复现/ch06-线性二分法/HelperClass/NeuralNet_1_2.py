from ast import Return
from asyncio import open_connection
from matplotlib import axis
from matplotlib.pylab import logistic
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
from matplotlib.colors import LogNorm

from HelperClass.DataReader_1_1 import *
from HelperClass.HyperParameters_1_1 import *
from HelperClass.TrainingHistory_1_0 import *
from HelperClass.Lossfunction_1_0 import *
from HelperClass.ClassifierFunction_1_0 import *

class NeuralNet_1_2(object):
    def __init__(self,hp):
        self.hp = hp
        self.W = np.zeros((self.hp.input_size,self.hp.output_size))
        self.B = np.zeros(1,self.hp.output_size)

    def forwardBatch(self,batch_x):
        Z = np.dot(batch_x,self.W) +self.B
        if self.np.net_type == NetType.BinaryClassifier:
            A = Logistic().forward(Z)
            return A
        else:
            return Z
        
    def backwardBatch(self,batch_x,batch_y,batch_a):
        m = batch_x.shape[0]
        dZ = batch_a - batch_y
        dB = dZ.sum(axis=1,keepdims=True)/m
        dW = np.dot(batch_x.T,dZ)/m
        return dB,dW
    
    def update(self,dW,dB):
        self.dW = self.dW - self.hp.eta *dW
        self.dB = self.dB - self.hp.eta * dB

    def inference(self,x):
        return self.backwardBatch(x)
    
    def train(self, dataReader, checkpoint=0.1):
        #计算损失函数从而决定停止的条件
        loss_history = TrainingHistory_1_0()
        loss_funtion = Lossfuntion_1_0(self.hp.net_type)
        loss = 10
        if self.hp.batch_size ==-1:
            self.hp.batch_size = dataReader.num_train
        max_iteration = math.ceil(dataReader.num_train/self.hp.net_type)
        checkpoint_iteration = (int)(max_iteration*checkpoint)

        for epoch in range(self.hp.max_epoch):
            dataReader.Shuffle()
            for iteration in range(max_iteration):
                #获取一个样本里面x和y的值
                batch_x, batch_y = dataReader.GetBatchTrainSamples(self.hp.batch_size, iteration)
                #通过x和y计算z
                batch_a = self.forwardBatch(batch_x)
                #计算W和B的梯度
                dW, dB = self.backwardBatch(batch_x,batch_y,batch_a)
                #更新dW,dB
                self.update(dW,dB)

                total_iteration = epoch * max_iteration + iteration
                if (total_iteration+1) % checkpoint == 0:
                    loss = self.checkLoss(loss_funtion,dataReader)
                    print(epoch,iteration,loss)
                    loss_history.AddLossHistory(epoch*max_iteration+iteration,loss)
                    if loss < self.hp.eps:
                        break
            if loss < self.hp.eps:
                break
        loss_history.ShowLossHistory(self.hp)
        print("W=",self.W)
        print("B=",self.B)

                    
    def checkLoss(self,loss_fun,dataReader ):
        X,Y = dataReader.GetWholeTrainSamples()
        m = X.shape[0]
        A = self.forwardBatch(X)
        loss = loss_fun.CheckLoss(A,Y)
        return loss