import matplotlib.pyplot as plt
from pulp import *#pulp 2.5.1
import matplotlib.colors as mcolors
import numpy as np
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

def ModeloRepoPuLP(SKU ,Ts ,T ,P ,C ,F ,SCD0 ,I0 ,Me ,Tr ,B ,Fvol ,semana):
    #combinaciones
    comb = [(i,j,t) for i in SKU for j in Ts for t in T]
    combSCD = [(i,t) for i in SKU for t in T]
    #crear modelo
    lp = LpProblem('RepoPuLP', LpMaximize)

    #agregar variables
    R    = LpVariable.dicts('R', comb, lowBound = 0, cat = 'Integer')
    Q    = LpVariable.dicts('Q', comb, lowBound = 0, cat = 'Integer')
    I    = LpVariable.dicts('I', comb, lowBound = 0, cat = 'Integer')
    SCD  = LpVariable.dicts('SCD', combSCD, lowBound = 0, cat = 'Integer')
    opt  = LpVariable.dicts('opt', comb,lowBound = None, cat = 'Binary')
    #funcion objetivo

    Gz1 = 0 #precio
    Gz2 = 1 #solo cantidad

    Gb1 = 1 #beneficio por evitar quiebres
    Gc1 = 1 #costo por tardar en vender

    z1 = (lpSum(Q[(i,j,t)] * (P[i,j][t] - C[i,j][t]) * Gz1 * (T[-1] - t)**2 for i,j,t in comb))
    z2 = (lpSum(Q[(i,j,t)] * Gz2 * (T[-1] - t)**2 for i,j,t in comb))

    B1 = (lpSum(opt[(i,j,t)] * F[i,j][t] * Gb1 * (T[-1] - t)**2 for i,j,t in comb))
    C1 = (lpSum(SCD[(i,t)] * Fvol[i] * Gc1 * t for i,t in combSCD))
    lp += z1 + z2 + B1 - C1
    #restricciones

    for i in SKU:
        for j in Ts:
            bigMinv = I0[i,j] + SCD0[i]
            for t in T:
                #opt=1:elijo elijo forecast
                #opt=0:elijo inventario
                #bigMs
                #bigMfc = 2 * F[i,j][t]
                bigMex = 2 * Me[i,j][t]
                #restricciones logicas
                lp += (Q[(i,j,t)] >= F[i,j][t] + bigMinv * (opt[(i,j,t)] - 1))
                lp += (Q[(i,j,t)] <= F[i,j][t])

                if t==semana:
                    #continuidad de inventario
                    lp += (R[(i,j,semana)] + I0[i,j] - Q[(i,j,semana)] - I[(i,j,semana)] == 0)
                    #restricciones logicas
                    lp += (Q[(i,j,semana)] <= I0[i,j] + R[(i,j,semana)])
                    #lp += (Q[(i,j,semana)] <= D[i,j][semana])
                    lp += (Q[(i,j,semana)] >= I0[i,j] + R[(i,j,semana)] - bigMinv*(opt[(i,j,semana)]))
                    #lp += (Q[(i,j,semana)] >= D[i,j][semana] - M*opt[(i,j,semana)])
                else:
                    #continuidad de inventario
                    lp += (R[(i,j,t)] + I[(i,j,t-1)] - Q[(i,j,t)] - I[(i,j,t)] == 0)
                    #restricciones logicas
                    lp += (Q[(i,j,t)] <= I[i,j,t-1] + R[(i,j,t)])
                    lp += (Q[(i,j,t)] >= (I[i,j,t-1] + R[(i,j,t)]) - bigMinv *(opt[(i,j,t)]))
                    lp += (I[(i,j,t-1)] >= Me[i,j][t] + bigMex * (opt[(i,j,t)] - 1))
    #no superar maximo almacenaje en tiendas
    for i in SKU:
        for t in T:
            if t==semana:
                lp += (lpSum(R[(i,j,semana)] for j in Ts) <= SCD0[i])
                lp += (SCD[(i,semana)] - SCD0[i] + lpSum(R[(i,j,semana)] for j in Ts) == 0)
            else:
                lp += (lpSum(R[(i,j,t)] for j in Ts) <= SCD[(i,t)])
                lp += (SCD[(i,t)] - SCD[(i,t-1)] + lpSum(R[(i,j,t)] for j in Ts) == 0)
    for j in Ts:
        for t in T:
            if t==semana:
                lp += (lpSum(R[(i,j,semana)] + I0[i,j] for i in SKU) <= B[j][semana])
            else:
                lp += (lpSum(R[(i,j,t)] + I[(i,j,t-1)] for i in SKU) <= B[j][t])
    #Restriccion limite de trasporte semanal
    for t in T:
        lp += (lpSum(R[(i,j,t)] * Fvol[i] for i in SKU for j in Ts) <= Tr[t])
    #Resolver el LP
    status = lp.solve(PULP_CBC_CMD(msg=0))
    #Imprimir la solución
    print('OPT = ', value(lp.objective))
    print()
    #guardo valor de los resultados de la optimizacion
    vals_repo  = { k : R[k].varValue for k in R }
    vals_inve  = { k : I[k].varValue for k in I }
    vals_venta = { k : Q[k].varValue for k in Q }
    vals_SCD   = { k : SCD[k].varValue for k in SCD }
    vals_opt   = { k : opt[k].varValue for k in opt }
    return {'R':vals_repo , 'I' : vals_inve ,'Q' : vals_venta , 'SCD' : vals_SCD ,'opt' : vals_opt}

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
#Mcostos     = np.array([[2990 ,3990],
#                        [10990 ,10990]])
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
Lcapacidad  = np.array([10000, 8000])

#inventario inicial periodo cero
Minv        = np.array([[0 ,0 ],
                        [0 ,0]])

#stock en centro de distribucion
SCD0 =  {1:100000,
        2:120000}

#obtengo curvas para utilizar
P       = CurvaPrecio(Mprecios ,fracciones ,semanas)
I0      = InventarioInicial(Minv)
F       = Demanda(Mdemanda ,Npeak , tamano ,Nsemanas)
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

'''
*****************************************************
                    Optimizacion
*****************************************************
'''
T = TT[:4]
semana = 1
model  = ModeloRepoPuLP(SKU ,Ts ,T ,P ,C ,F ,SCD0 ,I0 ,Me ,Tr ,B ,Fvol ,semana)

t2 = time.time()
print('tiempo de ejecucion: ',round(t2-t1,2))
