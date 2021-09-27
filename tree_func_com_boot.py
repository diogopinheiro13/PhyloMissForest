# -*- coding: utf-8 -*-
"""
@author: Diogo Pinheiro
Script that creates each decision tree for the bootstrap configuration.
"""
import numpy as np
import random


conta_ties = 0
# Here the user should define the decision tree parameters.
class Node:
    def __init__(self, x, y, idxs, min_leaf=0.13, max_depth = 1, max_features = 1, init_depth = -1):
        self.x = x 
        self.y = y
        self.idxs = idxs 
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.actual_depth = init_depth
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit()
        
    def find_varsplit(self):
        self.actual_depth = self.actual_depth + 1
         
        list_th, list_scores, list_variables, list_min, aux_min =[],[],[],[],[]
        aux_features = np.arange(0,self.col_count,1).tolist()
        aux_leaf = 0
        aux_leaf = int(self.col_count * self.min_leaf)
        self.min_leaf = aux_leaf
        if self.min_leaf < 1:
            self.min_leaf = 1
        
        if self.max_depth > 0:
            aux_depth = 0
            aux_depth = int(self.col_count * self.max_depth)
            self.max_depth = aux_depth
            
        if self.max_features == 1:
            conta = aux_features
        else:
            no_to_analyse = int(self.col_count * self.max_features)
            conta = random.sample(aux_features, no_to_analyse)
            
        for c in conta: 
            col, erro, point = self.find_better_split(c)
            if col == -1 and erro == -1 and point == -1:
                continue
            else:
                list_variables.append(col)
                list_scores.append(erro)
                list_th.append(point)
        
             
        if self.is_leaf == False:
            minimum = min([k for k in list_scores if k >= 0])
            aux_min = [index for index, value in enumerate(list_scores) if value == minimum]
            for i in range(0,len(aux_min)):
                list_min.append(list_variables[aux_min[i]])
        
        if all(v == -2 for v in list_variables) == True:
            self.score = float("inf")
        
        if self.is_leaf == False and self.score != float("inf") and len(list_min) > 0:
            best_idx = np.random.choice(list_min)
            variable = list_variables.index(best_idx)
            global conta_ties
            conta_ties = conta_ties + 1
            self.var_idx = list_variables[variable]
            self.score = list_scores[variable]
            self.split = list_th[variable]
            
        if self.is_leaf == True: 
            best_idx = np.random.choice(self.col_count)
            self.var_idx = best_idx
        
        if self.max_depth > 0:
            if self.actual_depth >= self.max_depth:
                self.score = float("inf")
     
        if self.is_leaf: return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        act = self.actual_depth
        self.lhs = Node(self.x, self.y, self.idxs[lhs], self.min_leaf, self.max_depth, self. max_features, act)
        self.rhs = Node(self.x, self.y, self.idxs[rhs], self.min_leaf, self.max_depth, self. max_features, act)
        
    def find_better_split(self, var_idx):
        col, score, point = -2,-2,-2
      
        x = self.x.values[self.idxs, var_idx]
        thresholds = np.unique(x)
        
        if len(thresholds) == 1 and len(x)>1:
            col = -1
            score = -1
            point = -1
            return col, score, point
                
        else:
            thresholds.sort()
            thresholds_list = []
            for th in range(0,len(thresholds)-1):
                mid_point = (thresholds[th]+thresholds[th+1])/2
                thresholds_list.append(mid_point)   
        
        for r in range(0,len(thresholds_list)):
            lhs = x <= thresholds_list[r]
            rhs = x > thresholds_list[r]
            
            if rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf: continue

            curr_score = self.find_score(lhs, rhs)
            if curr_score <= self.score: 
                self.var_idx = var_idx
                col = var_idx
                self.score = curr_score
                score = curr_score
                self.split = thresholds_list[r]
                point = thresholds_list[r]
                
        return col,score,point
    
    
    def find_score(self, lhs, rhs):
        y = self.y[self.idxs]
        lhs = y[lhs]
        rhs = y[rhs]
        
        left_m = lhs.mean()
        right_m = rhs.mean()
        
        aux_error = 0
        error_left = 0
        for i in range(0, len(lhs)):
            aux_error = ((left_m - lhs[i])**2)
            error_left = error_left + aux_error
        
        aux_error = 0   
        error_right = 0
        for i in range(0, len(rhs)):
            aux_error = ((right_m - rhs[i])**2)
            error_right = error_right + aux_error
        
        error = error_left + error_right
        return error
                
    @property
    def split_col(self): return self.x.values[self.idxs,self.var_idx]
                
    @property
    def is_leaf(self): return self.score == float('inf')                

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return node.predict_row(xi)
    

class DecisionTreeRegressor:
  
  def fit(self, X, y, min_leaf = 0.13):
    self.dtree = Node(X, y, np.array(np.arange(len(y))), min_leaf)
    global conta_ties
    retorno = conta_ties
    conta_ties = 0
    return self, retorno
  
  def predict(self, X):
    return self.dtree.predict(X.values)
