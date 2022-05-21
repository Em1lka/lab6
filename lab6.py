from colorsys import yiq_to_rgb
from re import X
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from math import floor
import numpy as np

class iMatrix:
    def __init__(self, k, n):
        self.k = k
        self.n = n
        self.matrixA = self.__createMatrixA()
        self.matrixF = np.copy(self.matrixA)
        self.printMatrixA()
        self.printMatrixF()
    
    def printMatrixA(self):
        print('Matrix A: ')
        print(self.matrixA)
        print('====================')
        
    def printMatrixF(self):
        print('Matrix F: ')
        print(self.matrixF)
        print('====================')
        
    def printResult(self):
        print('Result: ')
        print(self.result)
        print('====================')
    
    def __createMatrixA(self):
        return np.random.randint(-10, 11, size=(self.n, self.n)) # [-10; 10] == [-10; 11)
    
    def changeMatrixA(self, matrixA):
        self.matrixA = np.copy(matrixA)
        self.matrixF = np.copy(self.matrixA)
        self.printMatrixA()
        self.printMatrixF()
    
    def __checkCondition1(self):
        numEvenNumbers = 0 
        sumNumbers = 0
        for i in range(floor(self.n / 2), self.n):
            for j in range(floor(self.n / 2), self.n):
                if (j % 2 == 1 and self.matrixF[i][j] % 2 == 0):
                    numEvenNumbers+= 1
                if (i % 2 == 1):
                    sumNumbers += self.matrixF[i][j]  
        return (numEvenNumbers > sumNumbers)
                
    def __checkCondition2(self):
        detA = np.linalg.det(self.matrixA)
        sumDiagonalElements = np.trace(self.matrixF)
        return (detA > sumDiagonalElements)
            
    def __symmetricallySwapCE(self):
        for i in range(self.n): 
            for j in range(self.n): 
                if self.n-1-i < j and i >= floor(self.n / 2) and j >= floor(self.n / 2): 
                    self.matrixF[i][j],self.matrixF[self.n-1-j][self.n-1-i] = self.matrixF[self.n-1-j][self.n-1-i],self.matrixF[i][j]
        
    def __notSymmetricallySwapBE(self):
        for i in range(floor(self.n / 2)):
            for j in range(floor(self.n / 2)):
                temp = self.matrixF[0][0]
                np.append(self.matrixF[i], temp)
                np.delete(self.matrixF[i], 0)
    
    def changeMatrixF(self):
        if (self.__checkCondition1()):
            self.__symmetricallySwapCE()
            self.printMatrixF()
        else:
            self.__notSymmetricallySwapBE()
            self.printMatrixF()
        if(self.__checkCondition2()):
            self.result = self.matrixA * self.matrixA.transpose() - self.k * np.linalg.matrix_power(self.matrixF, -1)
        else:
            self.result = (np.linalg.matrix_power(self.matrixA, -1) + np.tril(self.matrixA) - np.linalg.matrix_power(self.matrixF, -1)) * self.k
    
try:    
    matrix = iMatrix(3, 3)
    matrix.changeMatrixF()
    matrix.printResult()
except:
    pass

print('==========LAB 6==========')
fig, ax = plt.subplots()                            #matplotlib
ax.set(xlabel='column number', ylabel='value')
for i in range(matrix.n):
    for j in range(matrix.n):
        plt.bar(i, matrix.result[i][j])
plt.show()

fig, ax = plt.subplots()
ax.set(xlabel='column number', ylabel='value')
ax.grid()
for j in range(matrix.n):
    ax.plot([i for i in range(matrix.n)], matrix.result[j][::])
plt.show()

ax = plt.figure().add_subplot(projection='3d')
ax.set(xlabel='x', ylabel='y', zlabel='z')
for i in range(matrix.n):
    plt.plot([j for j in range(matrix.n)], matrix.result[i][::], i)
plt.show()



sns.heatmap(data = matrix.result, annot = True)                 #seaborn
plt.xlabel('column number')
plt.ylabel('row number')
plt.show()

sns.boxplot(data = matrix.result)
plt.xlabel('column number')
plt.ylabel('value')
plt.show()

sns.lineplot(data = matrix.result)
plt.xlabel('column number')
plt.ylabel('value')
plt.show()