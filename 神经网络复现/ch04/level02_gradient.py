import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from HelperClass.DataReader_1_0 import *

file_name="ch04.npz"

if __name__ == '__main__':

    reader= DataReader_1_0(file_name)
    reader.ReadData()
    X,Y = reader.GetWholeTrainSamples()

    eta = 0.1
    w, b=0.0,0.0
    for i in range (reader.num_train):
        xi=X[i]
        yi=Y[i]

        zi = xi * w + b
        dz = zi - yi
        dw = dz * xi
        db = dz
        w = w - eta*dw
        b = b -eta*db
    
    print("w=",w)
    print("b=",b)