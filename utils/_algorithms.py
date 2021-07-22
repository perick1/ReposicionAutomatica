from pulp import *
import pyswarms as ps
import numpy as np
import random as rm

def pso6(Nsku,Nt,Ns,f,R,S,SC,IC,iter,Nparticles,PSO_params):
    #curvas (simulacion de pricing y compra d eproductos)
    CBT = curva6(R)
    CST = curva6(S)
    #crear bordes
    bounds =
    #argumentos de f
    kwargs={"A": a}
    #optimizador
    optimizer = ps.single.GlobalBestPSO(n_particles=Nparticles, dimensions=Nsku*Nt*Ns,bounds = bounds, options=PSO_params)
    cost, pos = optimizer.optimize(beneficio03, iters=iter, **kwargs)

    return optimizer
