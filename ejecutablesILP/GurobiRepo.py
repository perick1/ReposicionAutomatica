import numpy as np
from gurobipy import *
import os, sys

sys.path.append(os.path.abspath('C:/Users/Erick/Documents/GitHub/ReposicionAutomatica'))
from utils import *

R = {'R11':('escalon',30,15,3) ,'R12':('rampla',50,-10,3) ,'R13':('valle',15,10,15),
    'R21':('escalon',30,30,3) ,'R22':('rampla',50, -1,4) ,'R23':('valle',15,10,4)}
S = {'S01':('escalon',1000,0,1) ,'S02':('rampla',1000,1000,1) ,'S03':('valle',3000,3000,3000)}

model = Model('modelo 1')
#parametros
Sku = [1,2,3]
Tiendas = [1,2]
Semanas = [1,2,3,4,5,6]
#variables
vars = [(i,j,k) for i in Sku for j in Tiendas for k in Semanas]
x = model.addVars(vars ,vtype=GRB.CONTINUOUS,name='x')
#curvas
CBT = curva6(R,len(Sku),len(Tiendas)).reshape((len(Sku),len(Tiendas),len(Semanas)))#i,j,k
CST = curva6(S,len(Sku), 0)#j,k
#funcion objetivo
model.setObjective(quicksum(CBT[i-1,j-1,k-1]*x[i,j,k] for i in Sku for j in Tiendas for k in Semanas),GRB.MAXIMIZE)
#restricciones
for i in Sku:
    for k2 in Semanas:
        model.addConstr(quicksum(x[i,j,k] for j in Tiendas for k in range(1,k2+1))<=np.sum(CST[i-1,:k2-1]))
model.display()
model.optimize()
print('Funcion Objetivo: ',str(round(model.ObjVal,2)))
for v in model.getVars():
    print(str(v.VarName)+' = '+str(round(v.x,2)))
