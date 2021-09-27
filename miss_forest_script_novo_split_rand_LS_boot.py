# -*- coding: utf-8 -*-
"""
@author: Diogo Pinheiro
PhyloMissForest code for the bootstrap configuration.
Its using the optimal configuration found in the framework study.
Script that builds and uses the random forests to predict all the missing values in each distance matrix.
In the end, the completed matrices are returned.
"""

from tree_func_com_boot import DecisionTreeRegressor
import numpy as np
import pandas as pd


def check_best_ls_criteria(old,new,species):
    from bulding_tree import build_tree
    
    old_tree = build_tree(old.to_numpy(),species)
    new_tree = build_tree(new.to_numpy(),species)
    
    from least_square import least_squares_calc
    
    list_values = []
    
    list_values.append(least_squares_calc(old_tree, old, species))
    list_values.append(least_squares_calc(new_tree, new, species))
    
    minimum, idx = min((val, idx) for (idx, val) in enumerate(list_values))
    
    if idx == 0:
        return old,minimum
    else:
        return new,minimum


def turn_matrix_symetric(matrix):
    mat = matrix.copy()            
    for i in range(0,len(mat)):
        for j in range(0,len(mat)):
            if i != j:
                value = 0
                value = ((mat[i][j] + mat[j][i])/2)
                mat[i][j] = value
                mat[j][i] = value
            else:
                mat[i][j] = 0
    return mat

def get_upper_lower(matrix):
    lower = np.tril(matrix)
    upper = np.triu(matrix)
    lower = lower + lower.T - np.diag(np.diag(lower))
    upper = upper + upper.T - np.diag(np.diag(upper))
    matrix_aux = matrix.copy()  
    for i in range(0,len(matrix_aux)):
            for j in range(0,len(matrix_aux)):
                if i != j:
                    value = 0
                    value = ((matrix_aux[i][j] + matrix_aux[j][i])/2)
                    matrix_aux[i][j] = value
                    matrix_aux[j][i] = value
                else:
                    matrix_aux[i][j] = 0
    
    return lower, upper, matrix_aux

def check_best_min_evol(lower,upper,mean,species):
    from bulding_tree import build_tree
    
    l_tree = build_tree(lower,species)
    u_tree = build_tree(upper,species)
    m_tree = build_tree(mean.to_numpy(),species)
    
    from min_evolution import compute_minimum_evol
    
    list_values = []
    list_values.append(compute_minimum_evol(l_tree))
    list_values.append(compute_minimum_evol(u_tree))
    list_values.append(compute_minimum_evol(m_tree))
    
    minimum, idx = min((val, idx) for (idx, val) in enumerate(list_values))
    
    if idx == 0:
        return lower
    elif idx == 1:
        return upper
    else:
        return mean

def check_best_ls(lower,upper,mean,species):
    from bulding_tree import build_tree
    
    l_tree = build_tree(lower,species)
    u_tree = build_tree(upper,species)
    m_tree = build_tree(mean.to_numpy(),species)
    
    from least_square import least_squares_calc
    
    list_values = []
    
    list_values.append(least_squares_calc(l_tree, lower, species))
    list_values.append(least_squares_calc(u_tree, upper, species))
    list_values.append(least_squares_calc(m_tree, mean, species))
    
    
    minimum, idx = min((val, idx) for (idx, val) in enumerate(list_values))
    
    if idx == 0:
        return lower
    elif idx == 1:
        return upper
    else:
        return mean
    


def check_best_mat(lower,upper,mean, matrix_old):
    list_error =[]
    error = 0
    divisor = 0
    for i in range(0,len(matrix_old)):
        for j in range(0,len(matrix_old)):
            aux_error = 0
            aux_error = ((lower[i][j]-matrix_old[i][j]) ** 2)
            divisor = divisor + ((lower[i][j]) ** 2) 
            error = error + aux_error
    error_low = error/divisor
    
    list_error.append(error_low)
    
    error = 0
    divisor = 0
    for i in range(0,len(matrix_old)):
        for j in range(0,len(matrix_old)):
            aux_error = 0
            aux_error = ((upper[i][j]-matrix_old[i][j]) ** 2)
            divisor = divisor + ((upper[i][j]) ** 2) 
            error = error + aux_error
    error_up = error/divisor
    
    list_error.append(error_up)
    
    error = 0
    divisor = 0
    for i in range(0,len(matrix_old)):
        for j in range(0,len(matrix_old)):
            aux_error = 0
            aux_error = ((mean[i][j]-matrix_old[i][j]) ** 2)
            divisor = divisor + ((mean[i][j]) ** 2) 
            error = error + aux_error
    error_mean = error/divisor
    list_error.append(error_mean)
    
    menor = list_error.index(min(list_error))
    if menor == 0:
        return lower
    elif menor == 1:
        return upper
    else:
        return mean

def Q_matrix(dist_matrix):
    matrix_aux = dist_matrix.copy()            
    for i in range(0,len(dist_matrix)):
        for j in range(0,len(dist_matrix)):
            if i != j:
                value = 0
                value = ((matrix_aux[i][j] + matrix_aux[j][i])/2)
                matrix_aux[i][j] = value
                matrix_aux[j][i] = value
            else:
                matrix_aux[i][j] = 0
    sum_per_col = matrix_aux.sum(axis=1)
    
    Q = matrix_aux.copy()

    for i in range(0,len(dist_matrix)):
        for j in range(0,len(dist_matrix)):
            if i != j:
                Q[i][j] = ((len(matrix_aux)-2) * matrix_aux[i][j]) - sum_per_col[i] - sum_per_col[j]    
    return Q

def tie_break_rule(Q_matrix):
    tie_rule = []
    for i in range(0,len(Q_matrix)):
        act_col = Q_matrix[i]
        act_col = act_col.sort_values()
        tie_rule.append(act_col.index)
    return tie_rule  

def create_bootstrap_dataset(X_train,y_train,size_of_bootstrap):
    new_dim = round(len(y_train)*size_of_bootstrap)
    index = np.random.randint(0,len(y_train),new_dim)
    boot_X_train = pd.DataFrame(columns = X_train.columns)
    aux_list_boot_y = []
    for i in range(0,len(index)):
        boot_X_train = boot_X_train.append(X_train.iloc[index[i]],ignore_index = True)
        aux_list_boot_y.append(y_train.iloc[index[i]])
    boot_y_train = pd.DataFrame(aux_list_boot_y)
    return boot_X_train, boot_y_train   
    
def create_bootstrap_dataset(X_train,y_train,size_of_bootstrap):
    new_dim = round(len(y_train)*size_of_bootstrap)
    index = np.random.randint(0,len(y_train),new_dim)
    all_index = np.arange(0,len(y_train),1)
    
    boot_X_train = pd.DataFrame(columns = X_train.columns)
    out_of_bag_X = pd.DataFrame(columns = X_train.columns)
    
    #create the bootstrapped dataset
    aux_list_boot_y = []
    for i in range(0,len(index)):
        boot_X_train = boot_X_train.append(X_train.iloc[index[i]],ignore_index = True)
        aux_list_boot_y.append(y_train.iloc[index[i]])
    boot_y_train = pd.DataFrame(aux_list_boot_y)
    
    #colect the out of bag dataset
    aux_list_out_y = []
    for i in range(0,len(all_index)):
        if all_index[i] in index:
            continue
        else:
            out_of_bag_X = out_of_bag_X.append(X_train.iloc[all_index[i]],ignore_index = True)
            aux_list_out_y.append(y_train.iloc[all_index[i]])
    out_of_bag_y = pd.DataFrame(aux_list_out_y)
            
    return boot_X_train, boot_y_train, out_of_bag_X, out_of_bag_y, index

def random_forest_regressor(X_train,y_train,n_trees,bootstrap,bootstrap_size):
    trees_list = []
    ties_list = []
    if bootstrap == False:
        for tree in range(0,n_trees):
            y_train_copy = y_train.copy()
            y_train_copy = y_train_copy.to_numpy()
            regressor, no_of_ties = DecisionTreeRegressor().fit(X_train,y_train_copy)
            ties_list.append(no_of_ties)
            trees_list.append(regressor)
    else:
        list_x_out, list_y_out = [], []
        for tree in range(0,n_trees):
            boot_X_train, boot_y_train,  out_X, out_y, ind = create_bootstrap_dataset(X_train, y_train, bootstrap_size)
            list_x_out.append(out_X)
            list_y_out.append(out_y)
            y_train_boot_copy = boot_y_train.copy()
            y_train_boot_copy = y_train_boot_copy.to_numpy()
            regressor, no_of_ties = DecisionTreeRegressor().fit(boot_X_train,y_train_boot_copy)
            ties_list.append(no_of_ties)
            trees_list.append(regressor)

    return trees_list, ties_list

def create_mask(matrix_df):
    matrix_mask = matrix_df.isnull()
    return matrix_mask

def turn_matrix_symetric(matrix):
    mat = matrix.copy()            
    for i in range(0,len(mat)):
        for j in range(0,len(mat)):
            if i != j:
                value = 0
                value = ((mat[i][j] + mat[j][i])/2)
                mat[i][j] = value
                mat[j][i] = value
            else:
                mat[i][j] = 0
    return mat

def miss_forest_imputation(matrix,n_trees,bootstrap,bootstrap_size,act_spec):
    import time
    total_ties = 0
    start_time = time.time()
    #crete a boolean mask where the missing values are True and the known values are False
    mask = create_mask(matrix)
    
    #saving missing values positions
    missing_rows, missing_cols = np.where(mask)
    
    #count and sort the number of missing values per column
    miss_per_column = matrix.isna().sum().sort_values()
    index_order = miss_per_column.index
    
    #mean of each column for the known positions
    mean_col = matrix.mean(axis = 1, skipna = True) 
    
    #make an initial guess seting the missing values to the col mean
    for i in range(0,len(matrix)):
        for j in range(0,len(matrix)):
            if mask[i][j] == True:
                matrix[i][j] = mean_col[i]
    
    #turn the initial guess symetric
    matrix = turn_matrix_symetric(matrix)
           
    list_of_errors = []
    list_of_matrix = []
    new_error = 0
    old_error = np.inf
    #loop of imputation
    iteration = 0
    while new_error < old_error:
        print(iteration)
        #new iteration, which mean the old error needs to be updated
        if iteration > 0:
            old_error = new_error
            
        matrix_old = matrix.copy()
        
        #get the tie-break createria beased in the Neighbour Joining method
        Q = Q_matrix(matrix)
        rules_of_tie = tie_break_rule(Q)

        #second loop, impute column by column
        for column in index_order:
            
            #check if the column has missing values; if not skip this iteration
            no_of_miss = miss_per_column[column]
            if no_of_miss == 0:
                continue
            
            #split between candidate and non candidate columns
            candidate_col = matrix[column]
            other_col = matrix.drop(column,axis=1)
            
            X_train = other_col.copy()
            X_test = other_col.copy()
            y_train = candidate_col.copy()
            
            for i in range(0,len(mask)):
                if mask[i][column] == True:
                    X_train = X_train.drop(i,axis=0)
                    y_train = y_train.drop(i,axis=0)
                else:
                    X_test = X_test.drop(i,axis=0)
            
            if len(X_train) == 0 or len(y_train) == 0:
                continue
            
            #fitting a random forest
            random_for_est, ties_list = random_forest_regressor(X_train,y_train,n_trees,bootstrap,bootstrap_size)
            total_ties = total_ties + sum(ties_list)
            #predicting from the fitted random forest the values of the candidate column for X_test
            list_actual, all_X_test = [], []
            list_actual = []
            for tree_of_f in range(0,len(random_for_est)):  
                rgr = random_for_est[tree_of_f]
                act_prediction = rgr.predict(X_test)
                list_actual.append(act_prediction)
            
            #colecting the mean between all the predictions made by each tree
            y_pred_list = []
            for i in range(0,len(list_actual[0])):
                actual = 0
                final_val = 0
                for j in range(0, len(list_actual)):
                    actual = actual + list_actual[j][i]
                final_val = actual/len(random_for_est)
                y_pred_list.append(final_val)
            
            y_pred = pd.DataFrame(y_pred_list)
            y_pred.name = 'values'
            
            #impute the infered values in the matrix
            count = 0
            for i in range(0,len(matrix)):
                if mask[column][i] == True:
                    matrix[column][i] = y_pred.iloc[count]
                    count = count + 1
        
        lower, upper, mean = get_upper_lower(matrix)            
        new_mat = check_best_ls(lower,upper,mean,act_spec)
        new_mat = pd.DataFrame(new_mat)
        matrix = new_mat
        aux_mat,new_error = check_best_ls_criteria(matrix_old, matrix, act_spec)
        matrix = aux_mat
        list_of_errors.append(new_error)
        iteration = iteration + 1
        list_of_matrix.append(matrix_old)
            
    tempo =  time.time() - start_time   
    return matrix_old, tempo, iteration, total_ties