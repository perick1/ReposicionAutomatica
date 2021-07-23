# Import modules
import numpy as np
import matplotlib.pyplot as plt
import random as rm

def curva6(params_dict,Nsku,Nt):

    C = np.zeros((Nsku*Nt,len(params_dict)))
    for i in range(Nt):
        for j in range(Nsku):
            C[(i*Nsku)+j] = getCurve(params_dict[f'R{i+1}{j+1}'])
    return C

def getCurve(params_tuple):

    name ,p0 ,p1 ,p2 = params_tuple
    curve = np.zeros(6)

    for i in range(6):
        if name == 'escalon':
            curve[:p2] = p0
            curve[p2:] = p1
        elif name == 'rampla':
            curve[:p2] = p0
            curve[p2:] = np.arange(p0+p1,p0+p1*(7-p2),p1)
        elif name == 'valle':
            curve[0:2] = p0
            curve[2:4] = p1
            curve[4:6] = p2
    return curve
