# Import modules
import numpy as np
import matplotlib.pyplot as plt
import random as rm

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
