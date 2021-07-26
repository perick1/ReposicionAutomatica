from pulp import *
import numpy as np
import time
import random as rm

rm.seed(12)
np.random.seed(32)

def Combinations(n,m,l):
    L = []
    for i in range(n):
        for j in range(m):
            for k in range(l):
                L.append((i,j,k))
    return L

def getBounds(S,Nsku,Nt,Ns):
    s_max = np.resize(np.sum(S,axis = 1),Nsku*Nt)
    s_max = np.tile(s_max.reshape((Nsku*Nt,1)),(1,Ns))

    max_bound = s_max.reshape(Nsku*Nt*Ns)
    min_bound = max_bound * 0.0
    #return s_max
    return (min_bound ,  max_bound )

def getInitialConditions():
    pass
