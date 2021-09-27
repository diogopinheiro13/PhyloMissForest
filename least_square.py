# -*- coding: utf-8 -*-
"""
@author: Diogo Pinheiro
Function that given a phylogentic tree and the original distance matrix, calculates and returns the phylogenetic distance matrix.
"""
def least_squares_calc(tree,orig_mat,names):
    import numpy as np
    from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
    from Bio.Phylo.TreeConstruction import DistanceMatrix
    import ete3
    from ete3 import Tree
    from Bio import Phylo
    
    Phylo.write(tree, "aux_ls.nhx", 'newick')
    t1 = ete3.Tree("aux_ls.nhx",format=True)
    matrix_from_tree = np.zeros((len(names),len(names)))
    for i in range(0,len(names)):
        for j in range(0,len(names)):
            if i > j:
                matrix_from_tree[i][j] = t1.get_distance(names[i],names[j])
                matrix_from_tree[j][i] = t1.get_distance(names[i],names[j])
    
    least_squares_value_1 = 0.0
    least_squares_value_inverse = 0.0
    
    for i in range(0,len(names)):
        for j in range(0,len(names)):   
            if i > j:
                aux = (orig_mat[i][j] - matrix_from_tree[i][j])
                least_squares_value_1 = least_squares_value_1 + (aux ** 2)                 
    return least_squares_value_1