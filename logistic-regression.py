# -*- coding: utf-8 -*-
"""
Created on Tue May  8 14:58:04 2018

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt

def GetData(filename):
    datalen = len(open(filename).readline().split())-1
    file = open(filename)
    dataArr = []
    labelArr = []
    for line in file.readlines():
        tempArr =[1]
        item = line.strip().split()
        for i in range(datalen):
            tempArr.append(float(item[i]))
        dataArr.append(tempArr)
        labelArr.append(float(item[-1]))
    return dataArr,labelArr
        

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def GradentAscent(dataMat,labelMat):
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat).transpose()
    m,n = np.shape(dataMat)
    step = 0.001
    count=1
    weights = np.ones((n,1))
    while True:
        cost = labelMat - sigmoid(dataMat*weights)
        weights = weights + step*dataMat.T*cost
        count = count +1
        if(count>500):
            break
    return weights

def StoGradentAscent(dataMat,labelMat):
    dataMat = np.mat(dataMat)
    labelMat = labelMat
    m,n = np.shape(dataMat)
    step = 0.001
    count =1
    weights = np.mat(np.ones((n,1)))
    while count<400:
        for i in range(m):
            for j in range(n):
                h = sigmoid(np.sum(dataMat[i]*weights))
                cost = labelMat[i] - h
                weights = weights + step*cost*dataMat[i].T
        count+=1
    return weights

def figure(filename,weight):
    dataArr,labelArr = GetData(filename)
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    num = len(labelArr)
    for i in range(num):
        if(labelArr[i]==0):
            xcord1.append(dataArr[i][1])
            ycord1.append(dataArr[i][2])
        else:
            xcord2.append(dataArr[i][1])
            ycord2.append(dataArr[i][2])
    x= np.arange(-3,3,0.1)
    y= (-weight[0]-weight[1]*x)/weight[2]
    plt.plot(xcord1,ycord1,'ro',xcord2,ycord2,'gx')
    plt.plot(x,y.transpose())
    plt.show()        


def main():
     data,label = GetData('E:\\Data.txt')
     weight = StoGradentAscent(data,label)
     figure('E:\\Data.txt',weight)
     print(weight)

if __name__ =='__main__':
    main()