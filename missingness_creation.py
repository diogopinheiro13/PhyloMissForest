# -*- coding: utf-8 -*-
"""
@author: Diogo Pinheiro
Function that creates missing data in the inputted matrices
"""

def creat_missing_values(percentage,how_many,matrix):
    import numpy as np
    import random
    import math
    shape = matrix.shape
    no_of_entries = matrix.shape[0]*matrix.shape[0]
    list_of_matrix = []
    for i in range(0,how_many):
        booleans = np.zeros((matrix.shape[0],matrix.shape[0]))
        n=0
        infered = np.zeros((matrix.shape[0],matrix.shape[0]))
        
        while n < math.ceil((int(((shape[0]*shape[0])-shape[0])/2)*percentage)) : 
            x = random.randint(0, matrix.shape[0]-1)    
            y = random.randint(0, matrix.shape[0]-1)
            if x > y and booleans[x][y] == 0:
                infered[x][y] = float('nan')
                infered[y][x] = float('nan')
                booleans[x][y] = 1 
                n = n+1
        
        for k in range (0,matrix.shape[0]):
            for m in range(0,matrix.shape[0]):
                if k>=m and  booleans[k][m] == 0:
                    infered[k][m] = matrix [k][m]
                    infered[m][k] = matrix [k][m]
        list_of_matrix.append(infered)
    
    return list_of_matrix
        
            