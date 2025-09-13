import numpy as np

from HelperClass.DataReader_1_1 import *
file_name="ch05.npz"
if __name__=='__main__':
    reader = DataReader_1_1(file_name)
    reader.ReadData()
    X,Y =reader.GetWholeSample()
    num_example = X.shape[0]
    one = np.ones((num_example,1))
    x= np.column_stack((one,(X[0:num_example,:])))

    a = np.dot(x.T,x)
    b = np.asmatrix(a)
    c = np.linalg.inv(b)
    d = np.dot(c,x.T)
    e = np.dot(d,Y)
    print(e)
    b=e[0,0]
    w1=e[1,0]
    w2=e[2,0]
    print("b=",b)
    print("w1=",w1)
    print("w2=",w2)
    z = w1*15+w2*93+b
    print("z=",z)
