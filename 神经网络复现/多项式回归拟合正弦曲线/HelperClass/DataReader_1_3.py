"""
Version 1.3
what's new:
- add one-hot
"""

import numpy as np
from pathlib import Path


"""
X:
    x1:feature1, feature2,feature3......
    x2:....
Y:
    [if regression,value]
    [if binary classification, 0/1]
    [if multiple classification, eg 4 category, one-hot]
    
    
"""
class DataReader_1_3(object):
    def __init__(self,data_file):
        self.train_file_name = data_file
        self.num_train = 0
        self.XTrain = None #normalize x ,if not normalize, same as x raw
        self.YTrain = None #same above
        self.XRaw = None
        self.YRaw = None
        
    #read data from file
    def ReadData(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.load(self.train_file_name)
            self.XRaw = data['data']
            self.YRaw = data['label']
            self.num_train = self.XRaw.shape[0]
            self.XTrain = self.XRaw
            self.YTrain = self.YRaw
        else:
            raise Exception("Cannot find train file, please run ch09_1_data.py and ch09_2_data.py first.")
        #end if
        
    # normalize data by extracting range from source data
    # return: X_new :noromalize data with same shape
    # return: X_norm : N x 2
    #                 [[min1,range1]
    #                  [min2,range2]
    #                  [min3,range3]]
    def normalizeX(self):
        X_new = np.zeros(self.XRaw.shape)
        num_feature = self.XRaw.shape[1]
        self.X_norm = np.zeros((num_feature,2))
        # 按列归一化，即按照所有样本的同意特征值分别归一化
        for i in range (num_feature):
            #get one feature from all features 
            col_i = self.XRaw[:,i]
            max_value = np.max(col_i)
            min_value = np.min(col_i)
            #min value
            self.X_norm[i,0] = min_value
            #range value
            self.X_norm[i,1] = max_value - min_value
            new_col = (col_i - self.X_norm[i,0])/(self.X_norm[i,1])
            X_new[:,i] = new_col
        #end for
        self.XTrain = X_new
    
    #normalize data by self range and min_value
    def NormalizePredicateData(self,X_raw):
        X_new = np.zeros(X_raw.shape)
        n = X_raw.shape[1]
        for i in range (n):
            col_i = X_raw[:,1]
            X_new[:,1] = (col_i - self.X_norm[i,0]) / self.X_norm[i,1]
        return X_new
    
    def NormalizeY(self):
        self.Y_norm = np.zeros((1,2))
        max_value = np.max(self.YRaw)
        min_value = np.max(self.YRaw)
        self.Y_norm[0,0] = min_value
        self.Y_norm[0,1] = max_value - min_value
        y_new = (self.YRaw - min_value)/self.Y_norm[0,1]
        self.YTrain = y_new
    
    def ToOneHot(self,num_category,base=0):
        count = self.YRaw.shape[0]
        self.num_category = num_category
        y_new = np.zeros((count,self.num_category))
        for i in range(count):
            n = (int)(self.YRaw[i,0])
            y_new[i,n-base] = 1
        self.YTrain = y_new
    
    #get batch training data 
    def GetBatchTrainSample(self,iteration):
        x = self.XTrain[iteration]
        y = self.YTrain[iteration]
        return x,y

    #get batch training data 
    def GetBatchTrainSamples(self,batch_size,iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.XTrain[start:end,:]
        batch_Y = self.YTrain[start:end,:]
        return batch_X,batch_Y
    
    def GetWholeTrainSamples(self):
        return self.XTrain,self.YTrain
    
    def Shuffle(self):
        seed = np.random.randint(0,100)
        np.random.seed(seed)
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.YTrain)
        self.XTrain = XP
        self.YTrain = YP
    