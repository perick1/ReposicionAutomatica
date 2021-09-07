# Import modules
import numpy as np
import matplotlib.pyplot as plt
import random as rm
import pandas as pd
import os, sys
#from ReposicionAutomatica.utils import *
sys.path.append(os.path.abspath('C:/Users/Erick/Documents/GitHub/ReposicionAutomatica'))
from utils import *

Nsku = 3
Nt   = 2
Ns   = 6

best = [  284.55943137, 126.48331314, 363.80386733, 174.75853016,
          211.30114684,349.94858362, 14989.89399526,14058.83678499,
          13962.42732836, 14913.93787847,  7763.06679164,  5751.55980329,
          3543.70437993,  4296.86469865,  5790.94311242, 6105.27374078,
          6576.72790343, 14431.58074329,   517.05303277,   164.94274825,
          153.4653852,    632.33067201,   671.58105939,   581.46951039,
          13468.745088,   12685.751025,   10487.73725814,  7816.32398981,
          2015.46863075,  8066.5188279,  10931.51219395, 14179.07175166,
          1088.9419102, 3962.23898035, 7580.67027533,  2578.01915033]

col = ['semana 1','semana 2','semana 3','semana 4','semana 5','semana 6']
X = np.array(best)
Xdf = pd.DataFrame(X.reshape((Nt*Nsku,Ns)),columns = col)
Xdf['tienda'] =np.repeat(np.arange(1 ,Nt+1),Nsku)
Xdf['sku'] = np.resize(np.arange(1,Nsku+1),Nt*Nsku)
#Xdf.to_excel("output.xlsx")
#numpy.savetxt("best.csv", best, delimiter=",")



#R12: Revenue (beneficio) por venta en la tienda 1, del sku 2
curvas_BT   = {'R11':('escalon',30,15,3) ,'R12':('rampla',50,-10,3) ,'R13':('valle',15,10,15),
               'R21':('escalon',30,30,3) ,'R22':('rampla',50, -1,4) ,'R23':('valle',15,10,4)}
#S01: stock total a repartir del sku 1
curvas_ST   = {'S01':('escalon',1000,0,1) ,'S02':('rampla',0,1000,1) ,'S03':('valle',3000,3000,3000)}

restriccion_dentro_de_tienda   = {'IC10': 0.4 ,'IC20': 0.6}

plot_curvas(curvas_BT ,'dinero' ,Nsku ,Nt)      #curvas de beneficio (revenue)
plot_curvas(curvas_ST ,'stock' ,Nsku)      #curvas de stock a repartir
#x = PlotBest(mejor_pos,Nsku,Ntiendas,Nsemanas)
'''
def ConstrainsB(X ,R ,S ,params_tiendas ,Nsku ,Nt ,Ns):
    #penaltys
    CBT = curva6(R,Nsku,Nt)
    CST = curva6(S,Nsku, 0)

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

p1 ,p2 = ConstrainsB(X= X, R=curvas_BT ,S=curvas_ST,params_tiendas=restriccion_dentro_de_tienda ,Nsku=Nsku ,Nt=Nt ,Ns=Ns)
'''
