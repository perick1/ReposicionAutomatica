import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *
import time
'''
*****************************************************
                    Funciones
*****************************************************
'''
def CurvaPrecio(Mprecios ,fracciones ,semanas):
    dic = {}
    rows ,cols = Mprecios.shape
    for i in range(rows):
        for j in range(cols):
            precio_inicial = Mprecios[i,j]
            descuentos = np.array([])
            for s in range(len(semanas)):
                descuentos = np.concatenate((descuentos,np.ones(semanas[s]) * fracciones[s] * precio_inicial))
            key = (i+1 ,j+1)
            dic_semanas = { s+1 : int(descuentos[s]) for s in range(len(descuentos)) }
            dic[key] = dic_semanas
    return dic

def Costos(Mcostos ,Nsemanas):
    dic = {}
    rows ,cols = Mcostos.shape
    for i in range(rows):
        for j in range(cols):
            costo = Mcostos[i,j]
            key = (i+1 ,j+1)
            dic_semanas = { s+1 : int(costo) for s in range(Nsemanas) }
            dic[key] = dic_semanas
    return dic

def MinExhibicion(Mme ,Nsemanas):
    dic = {}
    rows ,cols = Mme.shape
    for i in range(rows):
        for j in range(cols):
            ME = Mme[i,j]
            key = (i+1 ,j+1)
            dic_semanas = { s+1 : int(ME) for s in range(Nsemanas) }
            dic[key] = dic_semanas
    return dic

def Demanda(Mdemanda ,Npeak , tamano ,Nsemanas):
    dic = {}
    rows ,cols = Mdemanda.shape
    for i in range(rows):
        for j in range(cols):
            Demanda     = Mme[i,j]
            key         = (i+1 ,j+1)
            dic_semanas = { s+1 : int(Demanda) for s in range(Nsemanas) }
            dic[key]    = dic_semanas

            dic_semanas[ Npeak ]   = int(Demanda * tamano)
            dic_semanas[Npeak + 1] = int(Demanda * (tamano +1) / 2)
            dic_semanas[Npeak - 1] = int(Demanda * (tamano +1) / 2)
    return dic

def Transporte(capacidad ,Nsemanas):
    dic = { s+1 : int(capacidad) for s in range(Nsemanas) }
    return dic

def StockTiendas(Lcapacidad ,Nsemanas):
    dic = {}
    Ntiendas = len(Lcapacidad)
    for j in range(Ntiendas):
        Cap = Lcapacidad[j]
        key = (j+1)
        dic_semanas = { s+1 : int(Cap) for s in range(Nsemanas) }
        dic[key] = dic_semanas
    return dic

def InventarioInicial(Minv):
    dic = {}
    rows ,cols = Minv.shape
    for i in range(rows):
        for j in range(cols):
            inv0 = Minv[i,j]
            key = (i+1 ,j+1)
            dic[key] = inv0
    return dic

def ModeloRepoGRB(SKU ,Ts ,T ,P ,C ,D ,SCD ,I0 ,Me ,Tr ,B ,F1 ,F2 ,Fvol ,semana):
    #combinaciones
    comb = [(i,j,t) for i in SKU for j in Ts for t in T]
    #crear modelo
    model = Model('Repo')
    model.Params.LogToConsole = 0
    #agregar variables
    R   = model.addVars(comb ,vtype=GRB.INTEGER ,name='R')
    V   = model.addVars(comb ,vtype=GRB.INTEGER ,name='V')
    I   = model.addVars(comb ,vtype=GRB.INTEGER ,name='I')
    opt = model.addVars(comb ,vtype=GRB.BINARY ,name='opt')
    #funcion objetivo
    z  = quicksum(V[i,j,t] * (P[i,j][t] - C[i,j][t]) for i in SKU for j in Ts for t in T)
    c1 = quicksum(I[i,j,t] * Fvol[i] * F1[t] for i in SKU for j in Ts for t in T)
    c2 = quicksum((SCD[i] -  quicksum(R[i,j,tau] for j in Ts for tau in T[:t])) * Fvol[i] * F2[t] for i in SKU for t in T)
    model.setObjective(z - c1 - c2 ,GRB.MAXIMIZE)

    #restricciones

    #reposicion no negativa
    model.addConstrs(R[i,j,t] >= 0 for i in SKU for j in Ts for t in T)
    #cumplir minimos de exhibicion
    #model.addConstrs((quicksum(Me[i,j2][t] for j2 in Ts) <= (SCD[i] -  quicksum(R[i,j2,tau] for j2 in Ts for tau in T[:t]))) >> (I[i,j,t] >= Me[i,j][t]) for i in SKU for j in Ts for t in T )
    model.addConstrs(I[i,j,t] >= Me[i,j][t] for i in SKU for j in Ts for t in T)
    #reparticion no supera el stock disponible en CdD
    model.addConstrs(quicksum(R[i,j,t] for j in Ts for t in T) <= SCD[i] for i in SKU)
    #no superar maximo almacenaje en tiendas
    model.addConstrs(quicksum(R[i,j,semana] + I0[i,j] for i in SKU) <= B[j][semana] for j in Ts)
    model.addConstrs(quicksum(R[i,j,t] + I[i,j,t-1] for i in SKU) <= B[j][t] for j in Ts for t in T if t!=semana)
    #continuidad de inventario
    model.addConstrs(R[i,j,semana] + I0[i,j] - V[i,j,semana] - I[i,j,semana] == 0 for i in SKU for j in Ts)
    model.addConstrs(R[i,j,t] + I[i,j,t-1] - V[i,j,t] - I[i,j,t] == 0 for i in SKU for j in Ts for t in T if t!=semana)
    #restricciones logicas
    model.addConstrs(V[i,j,semana] <= I0[i,j] + R[i,j,semana] for i in SKU for j in Ts)
    model.addConstrs(V[i,j,t] <= I[i,j,t-1] + R[i,j,t] for i in SKU for j in Ts for t in T if t!=semana)

    model.addConstrs(V[i,j,semana] <= D[i,j][semana] for i in SKU for j in Ts)
    model.addConstrs(V[i,j,t] <= D[i,j][t] for i in SKU for j in Ts for t in T if t!=semana)

    model.addConstrs((opt[i,j,semana] == 0) >> (V[i,j,semana] >= I0[i,j] + R[i,j,semana]) for i in SKU for j in Ts)
    model.addConstrs((opt[i,j,t] == 0) >> (V[i,j,t] >= I[i,j,t-1] + R[i,j,t]) for i in SKU for j in Ts for t in T if t!=semana)
    model.addConstrs((opt[i,j,semana] == 1) >> (V[i,j,semana] >= D[i,j][semana]) for i in SKU for j in Ts)
    model.addConstrs((opt[i,j,t] == 1) >> (V[i,j,t] >= D[i,j][t]) for i in SKU for j in Ts for t in T if t!=semana )

    #Restriccion limite de trasporte semanal
    model.addConstrs(quicksum(R[i,j,t] * Fvol[i] for i in SKU for j in Ts) <= Tr[t] for t in T)
    model.optimize()


    if T[0]==1:
        for v in model.getVars():
            print(str(v.VarName)+' = '+str(round(v.x,2)))

'''
*****************************************************
                    Parametros
*****************************************************
'''
t1 = time.time()
#precios
fracciones  = np.array([1.0 ,0.8 ,0.7 ,0.5 ,0.3])
semanas     = np.array([6 ,2 ,2 ,2 ,1])
Mprecios    = np.array([[2990 ,3990],
                        [10990 ,10990]])

#costos
Mcostos     = Mprecios * 0.3
Nsemanas    = np.sum(semanas)

#minimo exhibicion
Mme         = np.array([[100 ,50 ],
                        [50  ,40 ]])

#demanda
Mdemanda    = np.array([[300 ,300 ],
                        [150 ,100]])
Npeak       = 6
tamano      = 4

#trasporte
capacidad   = 1500

#stock conjunto maximo en tiendas
Lcapacidad  = np.array([1000, 800])

#inventario inicial periodo cero
Minv        = np.array([[0 ,0 ],
                        [0 ,0]])

#stock en centro de distribucion
SCD =  {1:100000,
        2:120000}

#obtengo curvas para utilizar
P       = CurvaPrecio(Mprecios ,fracciones ,semanas)
I0      = InventarioInicial(Minv)
D       = Demanda(Mdemanda ,Npeak , tamano ,Nsemanas)
C       = Costos(Mcostos ,Nsemanas)
Me      = MinExhibicion(Mme ,Nsemanas)
Tr      = Transporte(capacidad ,Nsemanas)
B       = StockTiendas(Lcapacidad ,Nsemanas)

#tiendas y skus
SKU = [1 ,2]
Ts  = [1 ,2]
#tiempo total en semanas
TT = [ t+1 for t in range(13)]
#otros costos
Fvol = {i:1 for i in SKU}
F1   = {t:0*t for t in TT}#en tienda
F2   = {t:0*t for t in TT}#scd

'''
*****************************************************
                    Optimizacion
*****************************************************
'''
for sem in TT[:-3]:
    T   = [sem ,sem+1 ,sem+2 ,sem+3]
    print(T)
    #optimizar 4 semans
    ModeloRepoGRB(SKU ,Ts ,T ,P ,C ,D ,SCD ,I0 ,Me ,Tr ,B ,F1 ,F2 ,Fvol ,sem)
    #actualizar valores de inventario y SCD

    #Obtener valores para graficar

'''
*****************************************************
                    Graficos
*****************************************************
'''
t2 = time.time()
print('tiempo de ejecucion: ',round(t2-t1,2))
