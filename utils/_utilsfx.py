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
