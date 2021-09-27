# PhyloMissForest


When running the script, the user will be prompted about how many datasets he wants to run and also their names. Those files should be .txt files and should be in the same folder as the source code. The first line of the file should containt the number of OTUs in the study, followed by the matrix.

This is an example of the content of the .txt file, where we have 4 OTUs and species names A, B, C and D:

4
A 0 1 2 3 
B 1 0 4 5
C 2 4 0 6
D 3 5 6 0 

Please note that this example is merely illustrative.

The percentages of missing data can be set in the array "percentages". The user can use as many percentages as he which.

The number of matrices per percentage of missing data is defined in the variable "times".

The number of times each matrix is imputted is equal to the variable "times_per_file".

The user should define the number of trees he want by setting the variable "number of trees".

When using bootstrap configuration, the user should set the variable "bootstrap_size".

In the begining of the tree_func_com_boot and tree_func_com non, the user should define the following parameters min_leaf, max_depth, max_features and init_depth.

The first one builds decision trees for the bootstrap configuration, while the second one does it for the non bootstrap definition.

Note that all those parameters are set according to the optimal configuration obtained during our study.

In the end the user should collect the NRF results plus the running time for each matrix. To facilitate this task, all information is processed and presented in a DataFrame called "result".



