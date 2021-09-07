# Import modules
import numpy as np
import matplotlib.pyplot as plt
import random as rm
import pandas as pd
import os, sys
#from ReposicionAutomatica.utils import *
sys.path.append(os.path.abspath('C:/Users/Erick/Documents/GitHub/ReposicionAutomatica'))
from utils import *

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)

np.random.seed(32)
rm.seed(12)

Nsku        = 3
Ntiendas    = 2
Nsemanas    = 6
funcion     = 'beneficio03'

#R12: Revenue (beneficio) por venta en la tienda 1, del sku 2
curvas_BT   = {'R11':('escalon',30,15,3) ,'R12':('rampla',50,-10,3) ,'R13':('valle',15,10,15),
               'R21':('escalon',30,30,3) ,'R22':('rampla',50, -1,4) ,'R23':('valle',15,10,4)}
#S01: stock total a repartir del sku 1
curvas_ST   = {'S01':('escalon',1000,0,1) ,'S02':('rampla',1000,1000,1) ,'S03':('valle',3000,3000,3000)}

#plot_curvas(curvas_BT ,'dinero' ,Nsku ,Ntiendas)      #curvas de beneficio (revenue)
#plot_curvas(curvas_ST ,'stock' ,Nsku)      #curvas de stock a repartir

#SC10: stock constrain (restriccion de stock total), distribucion a tiendas, fraccion del total de stock que va a la tienda 1
#IC10: (Inside constrain) restriccion dentro de la tienda 1, da maximo de la fraccion de la tienda que se puede albergar 1 solo sku
#restriccion_del_total_de_stock = {'SC10': 0.6 ,'SC20': 0.4}
restriccion_dentro_de_tienda   = {'IC10': 0.4 ,'IC20': 0.6}

Niteraciones = 20
Nparticulas  = 200
PSO_params   = {'c1': 1.0, 'c2': 1.0, 'w':1.5}

#hacer pso por 6 semanas, devuelve el objeto optimizador ya optimizado
PSO ,costo ,mejor_pos = pso6(Nsku=Nsku ,Nt=Ntiendas ,Ns=Nsemanas ,f=funcion ,R=curvas_BT ,S=curvas_ST,IC=restriccion_dentro_de_tienda ,iter=Niteraciones,Nparticles = Nparticulas ,PSO_params=PSO_params)
plot_cost_history(cost_history=PSO.cost_history)
plt.show(block=False)

#col = ['semana 1','semana 2','semana 3','semana 4','semana 5','semana 6']

#X = np.array(mejor_pos)
#Xdf = pd.DataFrame(X.reshape((Nttiendas*Nsku,Nsemanas)),columns = col)
#Xdf['tienda'] =np.repeat(np.arange(1 ,Ntiendas+1),Nsku)
#Xdf['sku'] = np.resize(np.arange(1,Nsku+1),Ntiendas*Nsku)
#Xdf.to_excel("output1.xlsx")
#x = PlotBest(mejor_pos,Nsku,Ntiendas,Nsemanas)

def ConstrainsB(X ,R ,S ,params_tiendas ,Nsku ,Nt ,Ns):
    #penaltys
    CBT = curva6(R,Nsku,Nt)
    CST = curva6(S,Nsku, 0)

    penalty1= 0
    penalty2= 0

    col = ['semana 1','semana 2','semana 3','semana 4','semana 5','semana 6']
    Rdf = pd.DataFrame(CBT,columns = col)
    Rdf['tienda'] = np.repeat(np.arange(1 ,Nt+1),Nsku)
    Rdf['sku'] = np.resize(np.arange(1,Nsku+1),Nt*Nsku)
    Sdf = pd.DataFrame(CST,columns = col)
    Sdf['sku'] = np.arange(1,Nsku+1)

    Xdf = pd.DataFrame(X.reshape((Nt*Nsku,Ns)),columns = col)
    Xdf['tienda'] =np.repeat(np.arange(1 ,Nt+1),Nsku)
    Xdf['sku'] = np.resize(np.arange(1,Nsku+1),Nt*Nsku)

    #computo penalizacion 1
    for i in range(1,Nsku+1):
        cantidad_de_venta = np.sum(Xdf.loc[Xdf['sku']==i].values ,axis = 0)[:-2]
        stock_acumulado = np.cumsum(Sdf.loc[Sdf['sku']==i].values)[:-1]
        penalty1 = penalty1 + np.sum((cantidad_de_venta > stock_acumulado))

    #computo penalizacion 2
    for j in range(1,Nt+1):
        max_repo = params_tiendas[f'IC{j}0']
        maxrepo_en_tienda = np.max(Xdf.loc[Xdf['tienda']==j].values,axis = 0)[:-2]
        penalty2 = penalty2 + np.sum((maxrepo_en_tienda > max_repo))

    return [penalty1 ,penalty2]

p1 ,p2 = ConstrainsB(X= mejor_pos, R=curvas_BT ,S=curvas_ST,params_tiendas=restriccion_dentro_de_tienda ,Nsku=Nsku ,Nt=Ntiendas ,Ns=Nsemanas)
