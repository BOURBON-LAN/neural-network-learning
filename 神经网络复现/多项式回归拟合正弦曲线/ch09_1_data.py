import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


train_data_name = "ch09_1.train.npz"
test_data_name = "ch09_1.test.npz"

def Targetfunction(x):
    p1 = np.sin(6.28*x)
    y = p1
    return y

def CreateSampleData(num_train,num_test):
    #create train data
    x1 = np.random.random((num_train,1))
    y1 = Targetfunction(x1) + (np.random.random((num_train,1))-0.5)/5
    np.savez(train_data_name, data=x1, label=y1)
    
    #create sample data
    x2 = np.linspace(0,1,num_test).reshape(num_test,1)
    y2 = Targetfunction(x2)
    np.savez(test_data_name,data=x2,label=y2)

def GetSampleData():
    Train_file = Path(train_data_name)
    Test_file = Path(test_data_name)
    if Train_file.exists() & Test_file.exists():
        Train_file = np.load(Train_file)
        Test_file = np.load(Test_file)
        return Train_file,Test_file

if __name__ == '__main__':
    CreateSampleData(500,100)
    TrainData, TestData = GetSampleData()
    plt.scatter(TrainData['data'],TrainData['label'],1,'g')
    plt.scatter(TestData["data"], TestData["label"], s=4, c='r')
    plt.show()