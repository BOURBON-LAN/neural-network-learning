import numpy as np
from pathlib import Path

class DataReader_1_1(object):
    def __init__(self, data_file):
        self.train_file_name = data_file
        self.num_train = 0
        self.XTrain = None  # normalized x, if not normalized, same as YRaw
        self.YTrain = None  # normalized y, if not normalized, same as YRaw
        self.XRaw = None    # raw x
        self.YRaw = None    # raw y
    
    #read data
    def ReadData(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.load(self.train_file_name)
            self.XRaw = data["data"]
            self.YRaw = data["label"]
            self.num_train = self.XRaw.shape[0]
            self.XTrain = self.XRaw
            self.YTrain = self.YRaw
        else:
            raise Exception("Cannot find train file, plz run ch05_data.py first")
        #  通过源数据中提取范围来规范化数据的

    def NormalizeX(self):
        X_new = np.zeros(self.XRaw.shape)
        num_feature = self.XRaw.shape[1]
        self.X_norm = np.zeros((num_feature,2))
        for i in range(num_feature):
            col_i = self.XRaw[:,i]
            max_value = np.max(col_i)
            min_value = np.min(col_i)
            self.X_norm[i,0]=min_value
            self.X_norm[i,1]=max_value - min_value
            new_col = (col_i - self.X_norm[i,0])/(self.X_norm[i,1])
            X_new[:,i] = new_col
        self.Xtrain = X_new

    def NormalizePredictData(self,X_raw):
        X_new = np.zeros(X_raw.shape())
        n = X_raw.shape[1]
        for i in range(n):
            col_i = X_raw[:,i]
            X_new[:,i] = (col_i-self.X_norm[i,0])/(self.X_norm[i,1])
        return X_new
    
    def NormalizeY(self):
        self.Y_norm = np.zeros((1,2))
        max_value = np.max(self.YRaw)
        min_value = np.min(self.YRaw)
        # min value
        self.Y_norm[0, 0] = min_value 
        # range value
        self.Y_norm[0, 1] = max_value - min_value 
        y_new = (self.YRaw - min_value) / self.Y_norm[0, 1]
        self.YTrain = y_new

    def GetSingleTrainSample(self,iteration):
        x = self.Xtrain[iteration]
        y = self.Ytrain[iteration]
        return x, y
    
    def GetBatchTrainSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.XTrain[start:end,:]
        batch_Y = self.YTrain[start:end,:]
        return batch_X, batch_Y
    
    def GetWholeSample(self):
        return self.Xtrain, self.Ytrain
    
    #机器学习中，训练集需要打乱顺序以保持X和Y 之间的关系不变，从而避免数据顺序影响
    #模型训练效果，比如避免批次训练中的偏差
    def Shuffle(self):
        seed = np.random.randint(0,100)
        np.random.seed(seed)
        XP = np.random.permutation(self.Xtrain)
        np.random.seed(seed)
        YP = np.random.permutation
        self.Xtrain = XP
        self.Ytrain = YP
        