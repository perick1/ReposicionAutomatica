from pulp import *
import pyswarms as ps
import numpy as np
import pandas as pd
import random as rm

sys.path.append(os.path.abspath('C:/Users/Erick/Documents/GitHub/ReposicionAutomatica'))
from utils import *

def pso6(Nsku,Nt,Ns,f,R,S,IC,iter,Nparticles,PSO_params):
    #curvas (simulacion de pricing y compra d eproductos)
    CBT = curva6(R,Nsku,Nt)
    CST = curva6(S,Nsku, 0)
    #crear bordes
    bounds = getBounds(CST,Nsku,Nt,Ns)
    #condiciones iniciales
    CI = getInitialConditions(100,10,Nparticles, Nsku ,Nt ,Ns)
    #argumentos de f
    kwargs={'R':CBT ,'S':CST ,'params_tiendas':IC, 'Npart':Nparticles ,'Nsku':Nsku ,'Nt':Nt,'Ns':Ns}
    #optimizador
    optimizer = ps.single.GlobalBestPSO(n_particles=Nparticles, dimensions=Nsku*Nt*Ns,bounds = bounds, options=PSO_params,init_pos = CI)
    cost, pos = optimizer.optimize(beneficio03, iters=iter, **kwargs)

    return [optimizer,cost,pos]
