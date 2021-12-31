import numpy as np
import matplotlib.pyplot as plt
import time

Ntiendas = np.array([2,8,15,20,25, 30, 40, 50])
Nskus    = np.array([2,6,10,20,50,100,200,500])
T = 8

data_pulp = np.array([[ 0.06  , 0.26    , 0.28    , 0.41      ,0.47      ,0.57     ,   0.87    ,  1.25   ],
                      [ 0.12  ,  0.63   , 1.28    , 10.97     ,  2.30    , 5.3     ,   3.26    ,  -1     ],
                      [ 0.15  , 0.76    ,  2.27   , 2.53      ,  3.97    , 5.82    ,   7.55    ,  -1     ],
                      [ 0.42  ,  2.15   ,  5.31   , 7.19      ,  7.87    , 10.03   , 21.11     ,  40.03  ],
                      [ 1.11  ,  6.41   , 24.02   , 31.83     , -1       , 67.38   , 149.28    ,  -1     ],
                      [ 2.35  ,  21.63  ,  78.95  , 150.52    ,   -1     , -1      ,    -1     , -1      ],
                      [ 3.02  ,  124.08 ,   -1    ,    -1     , -1       , -1      , -1        , -1      ],
                      [ 10.81 ,  -1     ,  -1     , -1        , -1       , -1       , -1       ,  -1     ]])

data_grb  = np.array([[ 0.01  ,  0.07   ,   0.10    , 0.10    ,  0.13    ,  0.16   ,  0.22     ,  0.29  ],
                      [ 0.05  ,  0.15   ,    0.27   , 0.38    ,  0.45    ,   0.68  ,   0.88    ,   1.14 ],
                      [ 0.07  , 0.26    ,   0.48    ,   0.66  ,  0.90    ,   1.05  ,   1.61    ,   2.29 ],
                      [ 0.13  ,   0.52  ,     1.14  ,  1.64   ,  2.15    ,    2.79 ,    4.32   ,  6.01  ],
                      [ 0.38  , 1.59    ,  3.90     ,   8.66  ,  12.19   ,   13.84 ,   38.37   ,  27.12 ],
                      [ 0.76  ,   3.37  ,      9.89 , 27.90   ,   58.01  , 37.63   ,    114.90 ,  189.45],
                      [ 1.59  ,   20.19 ,   31.56   , 133.42  ,   196.84 , -1      , -1        , -1     ],
                      [ 8.37  ,  54.82  ,    153.99 , -1      ,   -1     , -1      ,  -1       ,  -1    ]])




x1 = np.array([4*Nskus[i]*Ntiendas[j]*T + Nskus[i]*T                         for i in range(len(Nskus)) for j in range(len(Ntiendas))])
x2 = np.array([7*Nskus[i]*Ntiendas[j]*T + 2*Nskus[i]*T + Ntiendas[j]*T + T   for i in range(len(Nskus)) for j in range(len(Ntiendas))])

y1 = np.array([data_grb[i,j]  for i in range(len(Nskus)) for j in range(len(Ntiendas))])
y2 = np.array([data_pulp[i,j] for i in range(len(Nskus)) for j in range(len(Ntiendas))])

index1 = y1>0
index2 = y2>0

fig, axs = plt.subplots(nrows = 1, ncols = 2,figsize = (9,4))

ax1,ax2 = axs.flat

ax1.scatter(x1[index1],y1[index1],alpha=0.3,color = 'teal')
ax1.set_xlabel('Número de restricciones')
ax1.set_ylabel('Tiempo de ejecución [s]')
fig.suptitle('Solver Gurobi')
ax2.scatter(x2[index1],y1[index1],alpha=0.3,color = 'teal')
ax2.set_xlabel('Número de restricciones')
ax2.set_ylabel('Tiempo de ejecución [s]')
#ax.set_title('')
#fig.legend()
plt.show(block=False)
fig.tight_layout()

fig, axs = plt.subplots(nrows = 1, ncols = 2,figsize = (9,4))

ax1,ax2 = axs.flat

ax1.scatter(x1[index2],y2[index2],alpha=0.3,color = 'darkorange')
ax1.set_xlabel('Número de variables')
ax1.set_ylabel('Tiempo de ejecución [s]')
#ax.set_title('')
fig.suptitle('Solver PuLP CBC')
ax2.scatter(x2[index2],y2[index2],alpha=0.3,color = 'darkorange')
ax2.set_xlabel('Número de restricciones')
ax2.set_ylabel('Tiempo de ejecución [s]')
#ax.set_title('')
#fig.legend()
plt.show(block=False)
fig.tight_layout()
