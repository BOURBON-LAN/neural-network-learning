from hmac import new
from pyclbr import Class
import re
from matplotlib.dates import num2date
from matplotlib.pylab import permutation
import numpy as np
from pathlib import Path

from pyparsing import col

class DataReader_1_1(object):
    def __init__(self,data_file):
        self.train_file_name = data_file
        self.num_train = 0
        self.XTrain = None
        self.YTrain = None
        self.XRaw = None
        self.YRaw = None

    def ReadData(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.load(self.train_file_name)
            self.XRaw = data["data"]
            self.YRaw = data["label"]
            self.num_train = self.XRaw.shape[0]
            self.XTrain = self.XRaw
            self.YTrain = self.YRaw
        else :
            raise Exception("找不到文件，请在原文件夹中运行ch_06.npz")
    def NormalizeX(self):
        X_new = np.zeros(self.XRaw.shape)
        num_feature = self.XRaw.shape[1]
        self.X_norm = np.zeros((num_feature,2))
        for i in range(num_feature):
            col_i = self.XRaw[:,i]
            max_value = np.max(col_i)
            min_value = np.min(col_i)
            self.X_norm[i,0] = min_value
            self.X_norm[i,1] = max_value-min_value
            new_col = (col_i - self.X_norm[i,0])/(self.X_norm[i,1])
            X_new[:,i]=new_col
        self.XTrain = X_new

    def NormalizePredicateData(self, X_raw):
        X_new = np.zeros(X_raw.shape)
        n = X_raw.shape[1]
        for i in range(n):
            col_i = X_raw[:,i]
            X_new[:,i] = (col_i - self.X_norm[i,0]) / self.X_norm[i,1]
        return X_new
    
    def NormalizeY(self):
        self.Y_norm = np.zeros((1,2))
        max_value = np.max(self.YRaw)
        min_value = np.min(self.YRaw)
        self.Y_norm[0,0] = min_value
        self.Y_norm[0,1] = max_value - min_value
        y_new = (self.YRaw - min_value)/self.Y_norm[0,1]
        self.YTrain = y_new

    def GetBatchTrainSamples(self,batch_size,iteration):
        start = iteration*batch_size
        end = start + batch_size
        batch_X = self.XTrain[start:end,:]
        batch_Y = self.YTrain[start:end,:]
        return batch_X,batch_Y
    
    def GetWholeTrainSamples(self):
        seed = np.random.randint(0,100)
        np.random.seed(seed)
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.YTrain)
        self.XTrain = XP
        self.YTrain = YP