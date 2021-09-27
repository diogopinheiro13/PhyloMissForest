# -*- coding: utf-8 -*-
"""
@author: Diogo Pinheiro
Function that given two phylogenetic trees, calculates and returns the robinson_foulds distance between them.
"""

def compute_robinson_foulds(tree1,tree2):
    import ete3
    from ete3 import Tree
    from Bio import Phylo
    
    Phylo.write(tree1, "aux_1.nhx", 'newick')
    Phylo.write(tree2, "aux_2.nhx", 'newick')
    
    t1 = ete3.Tree("aux_1.nhx",format=True) 
    t2 = ete3.Tree("aux_2.nhx",format=True)
    
    results_a = t1.robinson_foulds(t2,unrooted_trees=True)
    robinson_foulds = 0.5*results_a[0] 
    n_robinson_foulds = robinson_foulds/results_a[1]
    return robinson_foulds, n_robinson_foulds

