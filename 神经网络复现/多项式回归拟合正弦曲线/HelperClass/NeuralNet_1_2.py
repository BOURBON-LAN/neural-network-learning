import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
from matplotlib.colors import LogNorm

from HelperClass.DataReader_1_3 import*
from HelperClass.HyperParameters_1_1 import*
from HelperClass.LossFunction_1_1 import *
from HelperClass.ClassifierFunction_1_1 import *
from HelperClass.TrainingHistory_1_0 import *

class NeuralNet_1_2(object):
    def __init__(self,params):
        self.params = params
        self.W = np.zeros((self.params.input_size, self.params.output_size))
        self.B = np.zeros((1,self.params.output_size))
        
    def forwardBatch(self,batch_x):
        Z = np.dot(batch_x,self.W) + self.B
        if self.params.net_type ==NetType.BinaryClassifier:
            A = Logistic().forward(Z)
            return A
        elif self.params.net_type == NetType.MultipleClassifier:
            A = Softmax().forward(Z)
            return A
        else:
            return Z
        
    def backwardBatch(self,batch_x,batch_y,batch_a):
        m = batch_x.shape[0]
        dZ = batch_a - batch_y
        dB = dZ.sum(axis=0,keepdims=True)/m
        dW = np.dot(batch_x.T,dZ)/m
        return dW,dB
    
    def update(self,dW,dB):
        self.W = self.W - self.params.eta * dW
        self.B = self.B - self.params.eta * dB
        
    def inference(self,x):
        return self.forwardBatch(x)
    
    def train(self,dataReader,checkpoint=0.1):
        #calculate the loss to decide the stop condition
        loss_history = TrainingHistory_1_0()
        loss_function = LossFunction_1_1(self.params.net_type)
        loss = 10
        if self.params.batch_size == -1:
            self.params.batch_size = dataReader.num_train
        max_iteraition = math.ceil(dataReader.num_train / self.params.batch_size)
        checkpoint_iteration = (int)(max_iteraition * checkpoint)
        
        for epoch in range(self.params.max_epoch):
            print("epoch:%d" %epoch)
            dataReader.Shuffle()
            for iteration in range(max_iteraition):
                #get x and y value for one sample
                batch_x, batch_y = dataReader.GetBatchTrainSamples(self.params.batch_size, iteration)
                #get z from x y
                batch_a = self.forwardBatch(batch_x)
                #calculate the gradient of w and b
                dW,dB = self.backwardBatch(batch_x,batch_y,batch_a)
                #update the w and b
                self.update(dW,dB)
                
                total_iteration = epoch * max_iteraition + iteration
                if (total_iteration+1) % checkpoint_iteration == 0:
                    loss = self.checkLoss(loss_function,dataReader)
                    print(epoch,total_iteration,loss)
                    loss_history.AddLossHistory(epoch*max_iteraition + iteration, loss)
                    if loss < self.params.eps:
                        break
            #end for
            if loss < self.params.eps:
                break
        #end for
        loss_history.ShowLossHistory(self.params)
        print("W:", self.W)
        print("B:", self.B)
    
    def checkLoss(self,loss_fun,dataReader):
        X,Y = dataReader.GetWholeTrainSamples()
        m = X.shape[0]
        A = self.forwardBatch(X)
        loss = loss_fun.CheckLoss(A,Y)
        return loss