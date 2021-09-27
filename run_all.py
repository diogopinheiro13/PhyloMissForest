# -*- coding: utf-8 -*-
"""
@author: Diogo Pinheiro
This is the main script of our source code. Here the user select the files to be inputted and get the final result in the DataFrame called "result".
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

#Reading the files and get the matrices with the list of the OTUS's names
no_of_file = int(input("How many files?\n"))
list_of_original_matrix = []
list_of_species = []
for file in range(0,no_of_file):
    name_actual_file = str(input("What is the name of the file\n"))
    file = open(name_actual_file, "r")
    lines = file.readlines()
    no_of_species = int(lines[0])
    species_names = []
    distance_matrix = np.zeros((no_of_species,no_of_species))
    for i in range(1,no_of_species+1):
        actual_line = lines[i]
        splited_line = actual_line.split()
        species_names.append(splited_line[0])
        for j in range(1,len(splited_line)):
            distance_matrix[j-1][i-1] = splited_line[j]
            distance_matrix[i-1][j-1] = splited_line[j]
    list_of_original_matrix.append(distance_matrix)
    list_of_species.append(species_names)

#Missingness Creation
percentages = [0.3] #define the percentages of missing data 
times= 1 # define how many matrices you want per percentage of missing data
times_per_file = 1 # define how many time you want to run each matrix
from missingness_creation import creat_missing_values

list_of_incomplete_matrices = []
for i in range(0,no_of_file):
    for per in range(0,len(percentages)):
        actual_list = creat_missing_values(percentages[per],times,list_of_original_matrix[i])
        list_of_incomplete_matrices.append(actual_list)

# RANDOM FOREST IMPUTATION WITH NON BOOTSTRAP CONFIGURATION
from miss_forest_script_novo_split_rand_LS_non import miss_forest_imputation

n_trees= 30 #define the number of trees per forest
bootstrap = False
bootstrap_size = 1
iteration , actual_file , count = 0 , 0 , 0
final_imputed_mean ,final_imputation_error_mean,final_exec_time_list, final_list_loop, final_list_ties= [],[],[],[],[]

for rep in range(0,times_per_file):
    count = 0
    actual_file = 0
    k=0
    l=0
    print(rep)
    imputed_matrices_mean = []
    imputation_error_mean_list = []
    imputation_exec_time_list = []
    list_loop = []
    list_ties_all = []
    
        
    for k in range(0,len(list_of_incomplete_matrices)):
        mean_list = []
        exec_time_list = []
        error_mean_list = []
        list_iteration = []
        list_ties = []
        
        for l in range(0,times):
            print("iteration number:")
            print(iteration)
            iteration = iteration + 1
            actual_matrix = []
            actual_matrix = np.copy(list_of_incomplete_matrices[k][l])
            actual_matrix = pd.DataFrame(actual_matrix)
            act_spec = list_of_species[actual_file]
            
            #imputing the missing values
            infered,exec_time, n_of_loops, total_ties = miss_forest_imputation(actual_matrix,n_trees,bootstrap,bootstrap_size,act_spec)
            
            matrix_aux = infered.copy()            
            for i in range(0,len(matrix_aux)):
                for j in range(0,len(matrix_aux)):
                    if i != j:
                        value = 0
                        value = ((matrix_aux[i][j] + matrix_aux[j][i])/2)
                        matrix_aux[i][j] = value
                        matrix_aux[j][i] = value
                    else:
                        matrix_aux[i][j] = 0
            matrix_aux = pd.DataFrame.to_numpy(matrix_aux)
            
            #saving variables
            mean_list.append(matrix_aux)
            exec_time_list.append(exec_time)
            imputation_rmse_mean = np.sqrt(mean_squared_error(matrix_aux,list_of_original_matrix[actual_file]))   
            error_mean_list.append(imputation_rmse_mean)
            list_iteration.append(n_of_loops)
            list_ties.append(total_ties)
            
            
        count = count + 1
        if count == len(percentages):
            actual_file = actual_file + 1
            count = 0
            
        imputed_matrices_mean.append(mean_list)
        imputation_error_mean_list.append(error_mean_list)
        imputation_exec_time_list.append(exec_time_list)
        list_loop.append(list_iteration)
        list_ties_all.append(list_ties)
    
    final_imputed_mean.append(imputed_matrices_mean)
    final_imputation_error_mean.append(imputation_error_mean_list)
    final_exec_time_list.append(imputation_exec_time_list)
    final_list_loop.append(list_loop)
    final_list_ties.append(list_ties_all)   
    
    
    
# RANDOM FOREST IMPUTATION WITH NON BOOTSTRAP CONFIGURATION
from miss_forest_script_novo_split_rand_LS_boot import miss_forest_imputation

n_trees=50 #define the number of trees per forest
bootstrap = True
bootstrap_size = 1 #bootstrap is set to true, the define the size of the bootstrapped datasets
iteration , actual_file , count = 0 , 0 , 0
final_imputed_mean_boot ,final_imputation_error_mean_boot ,final_exec_time_list_boot, final_list_loop_boot, final_list_ties_boot= [],[],[],[], []

for rep in range(0,times_per_file):
    count = 0
    actual_file = 0
    k=0
    l=0
    print(rep)
    imputed_matrices_mean_boot = []
    imputation_error_mean_list_boot = []
    imputation_exec_time_list_boot = []
    list_loop_boot = []
    list_ties_all_boot = []
    
        
    for k in range(0,len(list_of_incomplete_matrices)):
        mean_list_boot = []
        exec_time_list_boot = []
        error_mean_list_boot = []
        list_iteration_boot = []
        list_ties_boot = []
        
        for l in range(0,times):
            print("iteration number:")
            print(iteration)
            iteration = iteration + 1
            actual_matrix_boot = []
            actual_matrix = np.copy(list_of_incomplete_matrices[k][l])
            actual_matrix = pd.DataFrame(actual_matrix)
            act_spec = list_of_species[actual_file]
            #imputing the missing values
            infered,exec_time, n_of_loops, total_ties = miss_forest_imputation(actual_matrix,n_trees,bootstrap,bootstrap_size,act_spec)
            
            matrix_aux = infered.copy()            
            for i in range(0,len(matrix_aux)):
                for j in range(0,len(matrix_aux)):
                    if i != j:
                        value = 0
                        value = ((matrix_aux[i][j] + matrix_aux[j][i])/2)
                        matrix_aux[i][j] = value
                        matrix_aux[j][i] = value
                    else:
                        matrix_aux[i][j] = 0
            matrix_aux = pd.DataFrame.to_numpy(matrix_aux)
            
            #saving variables
            mean_list_boot.append(matrix_aux)
            exec_time_list_boot.append(exec_time)
            imputation_rmse_mean = np.sqrt(mean_squared_error(matrix_aux,list_of_original_matrix[actual_file]))   
            error_mean_list_boot.append(imputation_rmse_mean)
            list_iteration_boot.append(n_of_loops)
            list_ties_boot.append(total_ties)
                 
        count = count + 1
        if count == len(percentages):
            actual_file = actual_file + 1
            count = 0
            
        imputed_matrices_mean_boot.append(mean_list_boot)
        imputation_error_mean_list_boot.append(error_mean_list_boot)
        imputation_exec_time_list_boot.append(exec_time_list_boot)
        list_loop_boot.append(list_iteration_boot)
        list_ties_all_boot.append(list_ties_boot)
    
    final_imputed_mean_boot.append(imputed_matrices_mean_boot)
    final_imputation_error_mean_boot.append(imputation_error_mean_list_boot)
    final_exec_time_list_boot.append(imputation_exec_time_list_boot)
    final_list_loop_boot.append(list_loop_boot)
    final_list_ties_boot.append(list_ties_all_boot)
    
    
# AUTOENDODER IMPUTATION
from autoencoder import autoencoder_imputation

iteration , actual_file , count = 0 , 0 , 0
final_auto_imputed_mean ,final_auto_exec_time_list= [],[]
for rep in range(0,times_per_file):
    count = 0
    actual_file = 0
    k=0
    l=0
    print(rep)
    imputed_auto_matrices_mean = []
    imputation_auto_error_mean_list = []
    imputation_auto_exec_time_list = []

        
    for k in range(0,len(list_of_incomplete_matrices)):
        mean_auto_list = []
        exec_time_auto_list = []
        error_auto_mean_list = []
        list_auto_iteration = []
        list_auto_ties = []
        
        for l in range(0,times):
            print("iteration number:")
            print(iteration)
            iteration = iteration + 1
            actual_matrix = []

            actual_matrix = np.copy(list_of_incomplete_matrices[k][l])
            actual_matrix = pd.DataFrame(actual_matrix)
            act_spec = list_of_species[actual_file]
            #imputing the missing values
            infered,exec_time = autoencoder_imputation(actual_matrix)
            
            #saving variables
            mean_auto_list.append(infered)
            exec_time_auto_list.append(exec_time)
            
        count = count + 1
        if count == len(percentages):
            actual_file = actual_file + 1
            count = 0
            
        imputed_auto_matrices_mean.append(mean_auto_list)
        imputation_auto_exec_time_list.append(exec_time_auto_list)
    
    final_auto_imputed_mean.append(imputed_auto_matrices_mean)
    final_auto_exec_time_list.append(imputation_auto_exec_time_list)


# MATRIX FACTORIZATION IMPUTATION
from MatrixFactorization import matrix_fact_imputation

iteration , actual_file , count = 0 , 0 , 0
final_fact_imputed_mean ,final_fact_exec_time_list= [],[]
for rep in range(0,times_per_file):
    count = 0
    actual_file = 0
    k=0
    l=0
    print(rep)
    imputed_fact_matrices_mean = []
    imputation_fact_exec_time_list = []
    
        
    for k in range(0,len(list_of_incomplete_matrices)):
        mean_fact_list = []
        exec_time_fact_list = []
        error_fact_mean_list = []

        for l in range(0,times):
            print("iteration number:")
            print(iteration)
            iteration = iteration + 1
            actual_matrix = []
            actual_matrix = np.copy(list_of_incomplete_matrices[k][l])
            act_spec = list_of_species[actual_file]
            #imputing the missing values
            infered,exec_time = matrix_fact_imputation(actual_matrix)
    
            #saving variables
            mean_fact_list.append(infered)
            exec_time_fact_list.append(exec_time)
                       
        count = count + 1
        if count == len(percentages):
            actual_file = actual_file + 1
            count = 0
            
        imputed_fact_matrices_mean.append(mean_fact_list)
        imputation_fact_exec_time_list.append(exec_time_fact_list)
    
    final_fact_imputed_mean.append(imputed_fact_matrices_mean)
    final_fact_exec_time_list.append(imputation_fact_exec_time_list)

    
#Some data manipulation in order to run everythig more easy    
final_miss_for_mean = []
final_miss_for_mean_boot = []
final_auto = []
final_fact = []
for ite in range(0,times_per_file):
    miss_for_mean = []
    miss_for_mean_boot = []
    auto_mean = []
    fact_mean = []
    imputed_matrices_mean = final_imputed_mean[ite]
    imputed_matrices_mean_boot = final_imputed_mean_boot[ite]
    imputed_matrices_auto = final_auto_imputed_mean[ite]
    imputed_matrices_fact = final_fact_imputed_mean[ite]
    for k in range(0,no_of_file):
        miss_for_mean.append(imputed_matrices_mean[k*len(percentages):len(percentages)+len(percentages)*k])
        miss_for_mean_boot.append(imputed_matrices_mean_boot[k*len(percentages):len(percentages)+len(percentages)*k])
        auto_mean.append(imputed_matrices_auto[k*len(percentages):len(percentages)+len(percentages)*k])
        fact_mean.append(imputed_matrices_fact[k*len(percentages):len(percentages)+len(percentages)*k])
    final_miss_for_mean.append(miss_for_mean)
    final_miss_for_mean_boot.append(miss_for_mean_boot)
    final_auto.append(auto_mean)
    final_fact.append(fact_mean)

'''Building phylogenetic trees using neighbor joining from the matrices imputed with the algorithms'''
from bulding_tree import build_tree
final_list_all_trees_mean = []
final_list_all_trees_mean_boot = []
final_list_all_trees_auto = []
final_list_all_trees_fact = []
for repet in range(0,times_per_file):
    list_all_trees_mean = []
    list_all_trees_mean_boot = []
    list_all_trees_mean_auto = []
    list_all_trees_mean_fact = []
    for i in range(0,no_of_file):
        list_actual_trees_mean = []
        list_actual_trees_mean_boot = []
        list_actual_trees_mean_auto = []
        list_actual_trees_mean_fact = []
        actual_file_mean = final_miss_for_mean[repet][i]
        actual_file_mean_boot = final_miss_for_mean_boot[repet][i]
        actual_file_auto = final_auto[repet][i]
        actual_file_fact = final_fact[repet][i]
        for perc in range(0,len(percentages)):
            list_trees_act_file_mean =[]
            list_trees_act_file_mean_boot =[]
            list_trees_act_file_mean_auto = []
            list_trees_act_file_mean_fact = []
            for k in range(0,len(actual_file_mean[0])):
                actual_tree_mean = build_tree(actual_file_mean[perc][k],list_of_species[i])
                actual_tree_mean_boot = build_tree(actual_file_mean_boot[perc][k],list_of_species[i])
                actual_tree_auto = build_tree(actual_file_auto[perc][k],list_of_species[i])
                actual_tree_fact = build_tree(actual_file_fact[perc][k],list_of_species[i])
                list_actual_trees_mean.append(actual_tree_mean)  
                list_actual_trees_mean_boot.append(actual_tree_mean_boot)  
                list_actual_trees_mean_auto.append(actual_tree_auto)
                list_actual_trees_mean_fact.append(actual_tree_fact)
            list_trees_act_file_mean.append(list_actual_trees_mean)
            list_trees_act_file_mean_boot.append(list_actual_trees_mean_boot)
            list_trees_act_file_mean_auto.append(list_actual_trees_mean_auto)
            list_trees_act_file_mean_fact.append(list_actual_trees_mean_fact)
        list_all_trees_mean.append(list_actual_trees_mean)
        list_all_trees_mean_boot.append(list_actual_trees_mean_boot)
        list_all_trees_mean_auto.append(list_actual_trees_mean_auto)
        list_all_trees_mean_fact.append(list_actual_trees_mean_fact)
    final_list_all_trees_mean.append(list_all_trees_mean)
    final_list_all_trees_mean_boot.append(list_all_trees_mean_boot)
    final_list_all_trees_auto.append(list_all_trees_mean_auto)
    final_list_all_trees_fact.append(list_all_trees_mean_fact)
    
    
'''Building phylogenetic trees of the original matrices'''
original_tree = []    
for orig in range(0,len(list_of_original_matrix)):
    actual_orig_tree = build_tree(list_of_original_matrix[orig],list_of_species[orig])
    original_tree.append(actual_orig_tree)

'''Robinson Foulds metric calculation'''       
from robinson_foulds import compute_robinson_foulds
list_of_all_rf_m , list_of_all_nrf_m = [], []
list_of_all_rf_boot , list_of_all_nrf_boot = [], []
list_of_all_rf_auto , list_of_all_nrf_auto = [], []
list_of_all_rf_fact , list_of_all_nrf_fact = [], []
for repeti in range(0,times_per_file):
    list_of_rf_m , list_of_nrf_m = [], []
    list_of_rf_boot , list_of_nrf_boot = [], []
    list_of_rf_auto , list_of_nrf_auto = [], []
    list_of_rf_fact , list_of_nrf_fact = [], []
    for k in range(0,no_of_file):
        orig_tree = original_tree[k]
        aux_rf_m , aux_nrf_m =[], []
        aux_rf_boot , aux_nrf_boot =[], []
        aux_rf_auto , aux_nrf_auto =[], []
        aux_rf_fact , aux_nrf_fact =[], []
        for n in range(0,len(list_all_trees_mean[k])):
            to_comp_mean = final_list_all_trees_mean[repeti][k][n]
            to_comp_boot = final_list_all_trees_mean_boot[repeti][k][n]
            to_comp_auto = final_list_all_trees_auto[repeti][k][n]
            to_comp_fact = final_list_all_trees_fact[repeti][k][n]
            act_rf_m,act_nrf_m = compute_robinson_foulds(orig_tree,to_comp_mean)
            act_rf_boot,act_nrf_boot = compute_robinson_foulds(orig_tree,to_comp_boot)
            act_rf_auto,act_nrf_auto = compute_robinson_foulds(orig_tree,to_comp_auto)
            act_rf_fact,act_nrf_fact = compute_robinson_foulds(orig_tree,to_comp_fact)
            aux_rf_m.append(act_rf_m)
            aux_nrf_m.append(act_nrf_m)
            aux_rf_boot.append(act_rf_boot)
            aux_nrf_boot.append(act_nrf_boot)
            aux_rf_auto.append(act_rf_auto)
            aux_nrf_auto.append(act_nrf_auto)
            aux_rf_fact.append(act_rf_fact)
            aux_nrf_fact.append(act_nrf_fact)
        list_of_rf_m.append(aux_rf_m)
        list_of_nrf_m.append(aux_nrf_m)
        list_of_rf_boot.append(aux_rf_boot)
        list_of_nrf_boot.append(aux_nrf_boot)
        list_of_rf_auto.append(aux_rf_auto)
        list_of_nrf_auto.append(aux_nrf_auto)
        list_of_rf_fact.append(aux_rf_fact)
        list_of_nrf_fact.append(aux_nrf_fact)         
    list_of_all_rf_m.append(list_of_rf_m)
    list_of_all_nrf_m.append(list_of_nrf_m)
    list_of_all_rf_boot.append(list_of_rf_boot)
    list_of_all_nrf_boot.append(list_of_nrf_boot)
    list_of_all_rf_auto.append(list_of_rf_auto)
    list_of_all_nrf_auto.append(list_of_nrf_auto)
    list_of_all_rf_fact.append(list_of_rf_fact)
    list_of_all_nrf_fact.append(list_of_nrf_fact)



final_per_file_nrf_m = []
final_per_file_nrf_boot = []
final_per_file_nrf_auto = []
final_per_file_nrf_fact = []
for i in range(0,no_of_file):
    aux_per_file_m = []
    aux_per_file_boot = []
    aux_per_file_auto = []
    aux_per_file_fact = []
    for j in range(0,len(percentages)*times):
        total_m , total_mean_m = 0, 0
        total_m_boot , total_mean_m_boot = 0, 0
        total_m_auto , total_mean_m_auto = 0, 0
        total_m_fact , total_mean_m_fact = 0, 0
        aux_list_m = []
        aux_list_boot = []
        aux_list_auto = []
        aux_list_fact = []
        for each in range(0,times_per_file):
            total_m = total_m + list_of_all_nrf_m[each][i][j]
            total_m_boot = total_m_boot + list_of_all_nrf_boot[each][i][j]
            total_m_auto = total_m_auto + list_of_all_nrf_auto[each][i][j]
            total_m_fact = total_m_fact + list_of_all_nrf_fact[each][i][j]
        total_mean_m = total_m/times_per_file
        total_mean_m_boot = total_m_boot/times_per_file
        total_mean_m_auto = total_m_auto/times_per_file
        total_mean_m_fact = total_m_fact/times_per_file
        aux_list_m.append(total_mean_m)
        aux_list_boot.append(total_mean_m_boot)
        aux_list_auto.append(total_mean_m_auto)
        aux_list_fact.append(total_mean_m_fact)
        aux_per_file_m.append(aux_list_m)
        aux_per_file_boot.append(aux_list_boot)
        aux_per_file_auto.append(aux_list_auto)
        aux_per_file_fact.append(aux_list_fact)
    final_per_file_nrf_m.append(aux_per_file_m)
    final_per_file_nrf_boot.append(aux_per_file_boot)
    final_per_file_nrf_auto.append(aux_per_file_auto)
    final_per_file_nrf_fact.append(aux_per_file_fact)
    
final_nrf_mean_m = []
final_nrf_mean_boot = []
final_nrf_mean_auto = []
final_nrf_mean_fact = []
for i in range(0,len(final_per_file_nrf_m)):
    actual_list_m = []
    actual_list_boot = []
    actual_list_auto = []
    actual_list_fact = []
    for j in range(0,len(final_per_file_nrf_m[0])):
        actual_list_m = actual_list_m + final_per_file_nrf_m[i][j]
        actual_list_boot = actual_list_boot + final_per_file_nrf_boot[i][j]
        actual_list_auto = actual_list_auto + final_per_file_nrf_auto[i][j]
        actual_list_fact = actual_list_fact + final_per_file_nrf_fact[i][j]
    final_nrf_mean_m.append(actual_list_m)
    final_nrf_mean_boot.append(actual_list_boot)
    final_nrf_mean_auto.append(actual_list_auto)
    final_nrf_mean_fact.append(actual_list_fact)

final_time = []
final_time_boot = []
final_time_auto = []
final_time_fact = []
final_rmse_m = []
final_loop = []
final_ties_list = []
for i in range(0,len(final_exec_time_list[0])):
    aux_total_line = []
    aux_total_boot = []
    aux_total_auto = []
    aux_total_fact = []
    aux_total_line_m = []
    aux_total_line_loop = []
    aux_total_line_tie = []
    for j in range(0,times):
        total = 0
        total_boot = 0
        total_auto = 0
        total_fact = 0
        total_m = 0
        total_loop = 0
        total_tie = 0
        aux_line = []
        aux_boot = []
        aux_auto = []
        aux_fact = []
        aux_line_m = []
        aux_line_loop = []
        aux_line_tie = []
        for k in range(0,times_per_file):
            total = total + final_exec_time_list[k][i][j]
            total_boot = total_boot + final_exec_time_list_boot[k][i][j]
            total_auto = total_auto + final_auto_exec_time_list[k][i][j]
            total_fact = total_fact + final_fact_exec_time_list[k][i][j]
            total_m = total_m + final_imputation_error_mean[k][i][j]
            total_loop = total_loop + final_list_loop[k][i][j]
            total_tie = total_tie + final_list_ties[k][i][j]
        total_med = total/times_per_file
        total_med_boot = total_boot/times_per_file
        total_med_auto = total_auto/times_per_file
        total_med_fact = total_fact/times_per_file
        total_med_m = total_m/times_per_file
        total_med_loop = total_loop/times_per_file
        total_med_tie = total_tie/times_per_file
        
        aux_line.append(total_med)
        aux_boot.append(total_med_boot)
        aux_auto.append(total_med_auto)
        aux_fact.append(total_med_fact)
        aux_line_m.append(total_med_m)
        aux_line_loop.append(total_med_loop)
        aux_line_tie.append(total_med_tie)
        total_med = 0
        total_med_boot = 0
        total_med_auto = 0
        total_med_fact = 0
        total_med_m = 0
        total_med_loop = 0
        total_med_tie = 0
        aux_total_line.append(aux_line)
        aux_total_boot.append(aux_boot)
        aux_total_auto.append(aux_auto)
        aux_total_fact.append(aux_fact)
        aux_total_line_m.append(aux_line_m)
        aux_total_line_loop.append(aux_line_loop)
        aux_total_line_tie.append(aux_line_tie)
    final_time.append(aux_total_line)
    final_time_boot.append(aux_total_boot)
    final_time_auto.append(aux_total_auto)
    final_time_fact.append(aux_total_fact)
    final_rmse_m.append(aux_total_line_m)
    final_loop.append(aux_total_line_loop)
    final_ties_list.append(aux_total_line_tie)

#final data manipulation
list_nrf, list_time, list_rmse, list_aux ,list_file, list_ite, list_tie = [], [],[], [], [], [], []
list_time_boot = []
list_nrf_boot = []
list_time_auto = []
list_nrf_auto = []
list_time_fact = []
list_nrf_fact = []
for i in range(0,len(final_time)):
    aux_time = final_time[i]
    aux_time_boot = final_time_boot[i]
    aux_time_auto = final_time_auto[i]
    aux_time_fact = final_time_fact[i]
    aux_rmse = final_rmse_m[i]
    aux_ite = final_loop[i]
    aux_tie = final_ties_list[i]
    for k in range(0,len(aux_time)):
        list_time.append(aux_time[k][0]) 
        list_time_boot.append(aux_time_boot[k][0])
        list_time_auto.append(aux_time_auto[k][0])
        list_time_fact.append(aux_time_fact[k][0])
        list_rmse.append(aux_rmse[k][0])
        list_ite.append(aux_ite[k][0])
        list_tie.append(aux_tie[k][0])
data1 = pd.DataFrame(list_time)
data2 = pd.DataFrame(list_rmse)
data6 = pd.DataFrame(list_ite)
data7 = pd.DataFrame(list_tie)
data8 = pd.DataFrame(list_time_auto)
data10 = pd.DataFrame(list_time_fact)
data12 = pd.DataFrame(list_time_boot)

for i in range(0,no_of_file):
    act_list = []
    act_list_boot = []
    act_list_auto = []
    act_list_fact = []
    act_list = final_nrf_mean_m[i]
    act_list_boot = final_nrf_mean_boot[i]
    act_list_auto = final_nrf_mean_auto[i]
    act_list_fact = final_nrf_mean_fact[i]
    list_nrf = list_nrf + act_list
    list_nrf_boot = list_nrf_boot + act_list_boot
    list_nrf_auto = list_nrf_auto + act_list_auto
    list_nrf_fact = list_nrf_fact + act_list_fact
    
data3 = pd.DataFrame(list_nrf)
data9 = pd.DataFrame(list_nrf_auto)
data11 = pd.DataFrame(list_nrf_fact)
data13 = pd.DataFrame(list_nrf_boot)

for i in range(0,no_of_file):
    for j in range(0,len(percentages)):
            for k in range(0,times):    
                list_aux.append(str(percentages[j]))
                list_file.append(str(i))
data4 = pd.DataFrame(list_aux)
data5 = pd.DataFrame(list_file)

result = pd.concat([data8, data10, data1, data12, data9, data11, data3, data13, data4, data5],axis=1)
result.columns = ["Time AE","Time MF","Time RF NON","Time RF BOOT","NRF AE","NRF MF", "NRF RF NON","NRF RF BOOT","Percentage","File"]

import pickle
with open("final_miss_forest.bin", "wb") as output:
    pickle.dump(result, output)
