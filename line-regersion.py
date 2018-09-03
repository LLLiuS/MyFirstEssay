# -*- coding: utf-8 -*-
"""
Created on Sun May  6 15:57:54 2018

@author: liukaka
"""

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    numfeat = len(open(filename).readline().split('\t'))-1
    dataMat=[]
    lableMat=[]
    file = open(filename)
    for line in file.readlines():
        arrline=[]
        currentline = line.strip().split('\t')
        for i in range(numfeat):
            arrline.append(float(currentline[i]))
        dataMat.append(arrline)
        lableMat.append(float(currentline[-1]))
    print(dataMat,lableMat)
    return dataMat,lableMat

def standregres(xArr,yArr):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    xTx = xMat.T*xMat
    if(np.linalg.det(xTx)==0):
        print('can\'t inverse')
        return
    w = xTx.I*(xMat.T*yMat)
    return w

def figure():
    x,y=loadDataSet('E:\\ex0.txt')
    w = standregres(x,y)
    xl=[]
    xcopy = np.mat(x).copy()
    yHat = xcopy*w
    for i in x:
        xl.append(i[-1])#去除X的横坐标
    plt.plot(xcopy[:,1],yHat,xl,y,'ro')
    plt.grid(True)
    plt.show()
    print(np.corrcoef(yHat.T,np.mat(y))) #显示相关性
    
    
    
figure()

    
