from pulp import *
import pyswarms as ps
import numpy as np
import pandas as pd
import random as rm

def curva6(params_dict,Nsku,Nt):
    if Nt == 0 :
        C = np.zeros((Nsku,6))
        for i in range(Nsku):
            C[i] = getCurve(params_dict[f'S0{i+1}'])
    else:
        C = np.zeros((Nsku*Nt,6))
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

def getBounds(S,Nsku,Nt,Ns):
    print(S)
    s_max = np.resize(np.sum(S,axis = 1),Nsku*Nt)
    s_max = np.tile(s_max.reshape((Nsku*Nt,1)),(1,Ns))

    max_bound = s_max.reshape(Nsku*Nt*Ns)
    min_bound = max_bound * 0.0
    #return s_max
    return (min_bound ,  max_bound )

def getInitialConditions():
    pass

def beneficio03(X ,R ,S ,params_tiendas ,Npart ,Nsku ,Nt ,Ns):
    #penaltys
    penalty1= np.zeros(Npart)
    penalty2= np.zeros(Npart)
    revenue = np.zeros(Npart)
    factor  = 10**4

    col = ['semana 1','semana 2','semana 3','semana 4','semana 5','semana 6']
    Rdf = pd.DataFrame(R,columns = col)
    Rdf = pd.DataFrame(R,columns = col)
    Rdf['tienda'] = np.repeat(np.arange(1 ,Nt+1),Nsku)
    Rdf['sku'] = np.resize(np.arange(1,Nsku+1),Nt*Nsku)
    Sdf = pd.DataFrame(S,columns = col)
    Sdf['sku'] = np.arange(1,Nsku+1)

    for p in range(Npart):
        #crear pansas de X (3 columnas de indices, particulas y tiendas y sku) R (tiendas y sku) S(solo sku)

        Xdf = pd.DataFrame(X[p].reshape((Nt*Nsku,Ns)),columns = col)
        Xdf['tienda'] =np.repeat(np.arange(1 ,Nt+1),Nsku)
        Xdf['sku'] = np.resize(np.arange(1,Nsku+1),Nt*Nsku)
        #computo beneficio
        revenue[p] = np.sum( Xdf.values * Rdf.values )

        #computo penalizacion 1
        for i in range(1,Nsku+1):
            cantidad_de_venta = np.sum(Xdf.loc[Xdf['sku']==i].values ,axis = 0)[:-2]
            stock_acumulado = np.cumsum(Sdf.loc[Sdf['sku']==i].values)[:-1]
            penalty1[p] = penalty1[p] + np.sum((cantidad_de_venta > stock_acumulado) * factor)

        #computo penalizacion 2
        for j in range(1,Nt+1):
            max_repo = params_tiendas[f'IC{j}0']
            maxrepo_en_tienda = np.max(Xdf.loc[Xdf['tienda']==j].values,axis = 0)[:-2]
            penalty2[p] = penalty2[p] + np.sum((maxrepo_en_tienda > max_repo) * factor)

    score = penalty1 + penalty2 - revenue
    return score

def plot_curvas(params_dict ,tipo ,Nsku ,Nt=1):
    fig, axs = plt.subplots(nrows = Nt , ncols = Nsku)
    t = np.arange(1,7)

    for i in range(Nt):
        for j in range(Nsku):
            if tipo == 'stock':
                key = f'S0{j+1}'
                C = getCurve(params_dict[key])
                ax = axs[j]
                ax.bar(t,C)
            else:
                ax = axs[i,j]
                key = f'R{i+1}{j+1}'
                C = getCurve(params_dict[key])
                ax.plot(t,C)
                ax.set_ylim(0,np.max(C)*2)
            ax.set_title(key)
            ax.grid()

def pso6(Nsku,Nt,Ns,f,R,S,IC,iter,Nparticles,PSO_params):
    #curvas (simulacion de pricing y compra d eproductos)
    CBT = curva6(R,Nsku,Nt)
    CST = curva6(S,Nsku, 0)
    #crear bordes
    bounds = getBounds(CST,Nsku,Nt,Ns)
    #condiciones iniciales
    #CI =
    #argumentos de f
    kwargs={'R':CBT ,'S':CST ,'params_tiendas':IC, 'Npart':Nparticles ,'Nsku':Nsku ,'Nt':Nt,'Ns':Ns}
    #optimizador
    optimizer = ps.single.GlobalBestPSO(n_particles=Nparticles, dimensions=Nsku*Nt*Ns,bounds = bounds, options=PSO_params)#,init_pos = CI)
    cost, pos = optimizer.optimize(beneficio03, iters=iter, **kwargs)

    return optimizer
