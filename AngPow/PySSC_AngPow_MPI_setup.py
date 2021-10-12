#AngPow setup file for running the AngPow rouutines on cluster
#We consider a cluster of Nn nodes and Np processes in each node 

import os
present_rep = os.getcwd()

#path to AngPow repertory
AngPow_path = present_rep + '/AngPow/AngPow/' #finishing with '/' 

#machinefile is the text file that stores the IP addresses of all the nodes in the cluster network
#Since AngPow in using open MP, each node must refer to one process in the machinefile, i.e. Np = 1 (see machinefile_example)
machinefile = present_rep + '/AngPow/machinefile'

#number of used node/machine, Nn should not exceed the number of line in the machinefile
Nn = 28
