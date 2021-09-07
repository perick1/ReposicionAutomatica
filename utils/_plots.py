# Import modules
import numpy as np
import matplotlib.pyplot as plt
import random as rm

import os, sys

sys.path.append(os.path.abspath('C:/Users/Erick/Documents/GitHub/ReposicionAutomatica'))
from utils import *

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

def PlotBest(x,Nsku,Nt,Ns):
    col = ['semana 1','semana 2','semana 3','semana 4','semana 5','semana 6']
    Xdf = pd.DataFrame(x.reshape((Nt*Nsku,Ns)),columns = col)
    Xdf['tienda'] =np.repeat(np.arange(1 ,Nt+1),Nsku)
    Xdf['sku'] = np.resize(np.arange(1,Nsku+1),Nt*Nsku)
    fig, axs = plt.subplots(nrows = Nt , ncols = Nsku)
    t = np.arange(1,7)

    for i in range(1,Nt+1):
        for j in range(1,Nsku+1):
            curva = Xdf[(Xdf.sku ==j)&(Xdf.tienda==i)].values[0,:-2]
            ax = axs[i-1,j-1]
            title = f'sku:{j}, tienda{i}'
            ax.plot(t,curva)
            ax.set_ylim(0,np.max(curva)*2)
            ax.set_title(title)
            ax.grid()
    plt.show(block=False)
    return Xdf
