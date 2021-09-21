import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

'''
*****************************************************
                    Parametros
*****************************************************
'''
T   = [1 ,2 ,3 ,4]
SKU = [1 ,2]
Ts  = [1 ,2]
#precios
P   =  {(1,1):{1:200*4 ,2:200*4 ,3:200*4 ,4:200*4},
        (1,2):{1:300*4 ,2:300*4 ,3:300*4 ,4:300*4},
        (2,1):{1:400*4 ,2:400*4 ,3:400*4 ,4:400*4},
        (2,2):{1:100*4 ,2:100*4 ,3:100*4 ,4:100*4}}
#costos
C   =  {(1,1):{1:200 ,2:200 ,3:200 ,4:200},
        (1,2):{1:300 ,2:300 ,3:300 ,4:300},
        (2,1):{1:400 ,2:400 ,3:400 ,4:400},
        (2,2):{1:100 ,2:100 ,3:100 ,4:100}}
#IVA
IVA =  {(1,1):{1:200*0.19 ,2:200*0.19 ,3:200*0.19 ,4:200*0.19},
        (1,2):{1:300*0.19 ,2:300*0.19 ,3:300*0.19 ,4:300*0.19},
        (2,1):{1:400*0.19 ,2:400*0.19 ,3:400*0.19 ,4:400*0.19},
        (2,2):{1:100*0.19 ,2:100*0.19 ,3:100*0.19 ,4:100*0.19}}
#Neto
N   = C.copy()
for i in SKU:
    for j in Ts:
        for t in T:
            N[(i,j)][t] = P[(i,j)][t] - C[(i,j)][t] - IVA[(i,j)][t]

#minimos de exhibicion
Me  =  {(1,1):{1:50 ,2:50 ,3:50 ,4:50},
        (1,2):{1:30 ,2:30 ,3:30 ,4:30},
        (2,1):{1:80 ,2:80 ,3:80 ,4:80},
        (2,2):{1:120 ,2:120 ,3:120 ,4:120}}
#stock conjunto en tienda
B   =  {1:{1:800  ,2:800  ,3:800  ,4:800},
        2:{1:1000 ,2:1000 ,3:1000 ,4:1000}}
#stock en centro de distribucion
SCD =  {1:1000,
        2:2000}
#inventario inicial periodo cero
I0  =  {(1,1):0,
        (1,2):0,
        (2,1):0,
        (2,2):0}
#demanda
D   =  {(1,1):{1:300 ,2:300 ,3:300 ,4:300},
        (1,2):{1:500 ,2:500 ,3:500 ,4:500},
        (2,1):{1:200 ,2:200 ,3:200 ,4:200},
        (2,2):{1:600 ,2:600 ,3:600 ,4:600}}

#combinaciones
comb = [(i,j,t) for i in SKU for j in Ts for t in T]

'''
*****************************************************
                    Modelo y variables
*****************************************************
'''
#crear modelo
model = Model('Repo')
#agregar variables
R   = model.addVars(comb ,vtype=GRB.INTEGER ,name='R')
V   = model.addVars(comb ,vtype=GRB.INTEGER ,name='V')
I   = model.addVars(comb ,vtype=GRB.INTEGER ,name='I')
opt = model.addVars(comb ,vtype=GRB.BINARY ,name='opt')
#funcion objetivo
z = quicksum(V[i,j,t] * N[i,j][t] for i in SKU for j in Ts for t in T)
model.setObjective(z,GRB.MAXIMIZE)

'''
*****************************************************
                    Restricciones
*****************************************************
'''
#reposicion no negativa
model.addConstrs(R[i,j,t] >= 0 for i in SKU for j in Ts for t in T)
#cumplir minimos de exhibicion
model.addConstrs(R[i,j,1] >= Me[i,j][1] - I0[i,j] for i in SKU for j in Ts)
model.addConstrs(R[i,j,t] >= Me[i,j][t] - I[i,j,t-1] for i in SKU for j in Ts for t in T if t!=1)
#reparticion no supera el stock disponible en CdD
model.addConstrs(quicksum(R[i,j,t] for j in Ts for t in T) <= SCD[i] for i in SKU)
#no superar maximo almacenaje en tiendas
model.addConstrs(quicksum(R[i,j,1] + I0[i,j] for i in SKU) <= B[j][1] for j in Ts)
model.addConstrs(quicksum(R[i,j,t] + I[i,j,t-1] for i in SKU) <= B[j][t] for j in Ts for t in T if t!=1)
#continuidad de inventario
model.addConstrs(R[i,j,1] + I0[i,j] - V[i,j,1] - I[i,j,1] == 0 for i in SKU for j in Ts)
model.addConstrs(R[i,j,t] + I[i,j,t-1] - V[i,j,t] - I[i,j,t] == 0 for i in SKU for j in Ts for t in T if t!=1)
#restricciones logicas
model.addConstrs(V[i,j,1] <= I0[i,j] + R[i,j,1] for i in SKU for j in Ts)
model.addConstrs(V[i,j,t] <= I[i,j,t-1] + R[i,j,t] for i in SKU for j in Ts for t in T if t!=1)

model.addConstrs(V[i,j,1] <= D[i,j][1] for i in SKU for j in Ts)
model.addConstrs(V[i,j,t] <= D[i,j][t] for i in SKU for j in Ts for t in T if t!=1)

model.addConstrs((opt[i,j,1] == 0) >> (V[i,j,1] >= I0[i,j] + R[i,j,1]) for i in SKU for j in Ts)
model.addConstrs((opt[i,j,1] == 0) >> (V[i,j,t] >= I[i,j,t-1] + R[i,j,t]) for i in SKU for j in Ts for t in T if t!=1)
model.addConstrs((opt[i,j,t] == 1) >> (V[i,j,1] >= D[i,j][1]) for i in SKU for j in Ts)
model.addConstrs((opt[i,j,t] == 1) >> (V[i,j,t] >= D[i,j][t]) for i in SKU for j in Ts for t in T if t!=1)

'''
*****************************************************
                    Resultados
*****************************************************
'''
#model.display()
model.optimize()
print('Funcion Objetivo: ',str(round(model.ObjVal,2)))
for v in model.getVars():
    print(str(v.VarName)+' = '+str(round(v.x,2)))
'''
*****************************************************
                    Graficos
*****************************************************
'''
sku = 2
tienda = 1

r    = [R[sku,tienda,t].x for t in T] + [0]
print(sum(r))
v    = [V[sku,tienda,t].x for t in T]
inv  = [I[sku,tienda,t].x for t in T]
inv  = [I0[sku,tienda]] + inv
d    = [D[sku,tienda][t] for t in T]
T2   = T + [5]


y_offset = np.zeros(5)
bar_width = 0.6

fig, ax = plt.subplots(figsize = (8,5))
ax.plot(T,d,ls = '-',color = 'red',label = 'demanda')
ax.plot(T,v,ls = '',marker = '*',color = 'blue',label = 'ventas')
ax.bar(T2, inv, bar_width, bottom=y_offset, color='gray',label = 'inventario')
y_offset = y_offset + np.array(inv)
ax.bar(T2, r, bar_width, bottom=y_offset, color='green',label = 'reposicion')
ax.set_title(f'SKU {sku} en tienda {tienda}')
ax.legend()
plt.show(block=False)

'''
*****************************************************

*****************************************************
'''
