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

def getInitialConditions(M1 ,M2, Npart ,Nsku ,Nt ,Ns):
    Rand = np.random.randint(M1, size = (Npart,Nsku *Nt *Ns))
    Rand[0] = np.random.randint(M2, size = Nsku *Nt *Ns)
    #InitCond = Rand.reshape((Nt*Nsku,Ns))

    return Rand.astype(float)

def ConstrainsB(X ,R ,S ,params_tiendas ,Npart ,Nsku ,Nt ,Ns):
    #penaltys
    penalty1= 0
    penalty2= 0

    col = ['semana 1','semana 2','semana 3','semana 4','semana 5','semana 6']
    Rdf = pd.DataFrame(R,columns = col)
    Rdf = pd.DataFrame(R,columns = col)
    Rdf['tienda'] = np.repeat(np.arange(1 ,Nt+1),Nsku)
    Rdf['sku'] = np.resize(np.arange(1,Nsku+1),Nt*Nsku)
    Sdf = pd.DataFrame(S,columns = col)
    Sdf['sku'] = np.arange(1,Nsku+1)

    Xdf = pd.DataFrame(X.reshape((Nt*Nsku,Ns)),columns = col)
    Xdf['tienda'] =np.repeat(np.arange(1 ,Nt+1),Nsku)
    Xdf['sku'] = np.resize(np.arange(1,Nsku+1),Nt*Nsku)
    #computo beneficio
    revenue[p] = np.sum( Xdf.values * Rdf.values )

    #computo penalizacion 1
    for i in range(1,Nsku+1):
        cantidad_de_venta = np.sum(Xdf.loc[Xdf['sku']==i].values ,axis = 0)[:-2]
        stock_acumulado = np.cumsum(Sdf.loc[Sdf['sku']==i].values)[:-1]
        penalty1 = penalty1 + np.sum((cantidad_de_venta > stock_acumulado) * factor)

    #computo penalizacion 2
    for j in range(1,Nt+1):
        max_repo = params_tiendas[f'IC{j}0']
        maxrepo_en_tienda = np.max(Xdf.loc[Xdf['tienda']==j].values,axis = 0)[:-2]
        penalty2 = penalty2 + np.sum((maxrepo_en_tienda > max_repo) * factor)

    return [penalty1 ,penalty2]
