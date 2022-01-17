import numpy as np
import matplotlib.pyplot as plt

semanas = np.arange(1,21)
demanda = np.ones(20)
demanda[7] =  4
demanda[[6,8]] = 2.5

fig, ax1 = plt.subplots(figsize = (8,3))


ax1.plot(semanas,demanda,alpha=0.8,color = 'teal',marker ='X',markersize = 15)
ax1.set_ylabel('Demanda normalizada')
ax1.set_xlabel('Semanas')
ax1.set_xlim(0.8,20.2)
ax1.set_ylim(-0.5,5.5)
ax1.annotate("",
            xy=(1.5, -0.49), xycoords='data',
            xytext=(1.5, 0.99), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3"),
            )
ax1.annotate("Demanda base",
            xy=(1.5, -0.49), xycoords='data',
            xytext=(1.6, 0.1), textcoords='data')

ax1.annotate("",
            xy=(8, -0.49), xycoords='data',
            xytext=(8, 3.89), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3"),
            )
ax1.annotate("peak de demanda",
            xy=(8, -0.49), xycoords='data',
            xytext=(8.1, 1.6), textcoords='data')
#ax2.set_xlim(-3000,62000)
ax1.set_xticks(semanas)
ax1.set_yticks([1,2.5,4])
#fig.suptitle('Solver Gurobi')
fig.suptitle('Curva general de demanda para cada SKU/tienda, normalizada a la demanda base')
#ax.set_title('')
#fig.legend()
ax1.grid(axis = 'both' ,color='gray', linestyle='--' , linewidth=0.5)
plt.show(block=False)
fig.tight_layout()

'''
precios = [1,1,1,1,1,1,0.8,0.8,0.8,0.7,0.7,0.7,0.5,0.5,0.5,0.3,0.3,0.3,0.3,0.3]

fig, ax1 = plt.subplots(figsize = (8,3))


ax1.plot(semanas,precios,alpha=0.8,color = 'orangered',marker ='X',markersize = 15)
ax1.set_ylabel('Precio normalizado')
ax1.set_xlabel('Semanas')
ax1.set_xlim(0.8,20.2)
ax1.set_ylim(-0.0,1.2)
#ax2.set_xlim(-3000,62000)
ax1.set_xticks(semanas)
ax1.set_yticks([0.3,0.5,0.7,0.8,1])
#fig.suptitle('Solver Gurobi')
fig.suptitle('Curva general de precio para cada SKU/tienda, normalizada al precio inicial')
#ax.set_title('')
#fig.legend()
ax1.grid(axis = 'both' ,color='gray', linestyle='--' , linewidth=0.5)
plt.show(block=False)
fig.tight_layout()
'''
