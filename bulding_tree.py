# -*- coding: utf-8 -*-
"""
@author: Diogo Pinheiro
Function that builds a phylogenetic tree from a distance matrix, using NJ method.
"""
def build_tree(matrix,names):
    from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
    from Bio.Phylo.TreeConstruction import DistanceMatrix
    
    mat = matrix.tolist()
    new_list = []
    for i in range(0, len(mat)):
        pick_element = mat[i]
        chose_values = pick_element[0:i+1]
        new_list.append(chose_values)
    
    m = DistanceMatrix(names, new_list)
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(m)
    return tree