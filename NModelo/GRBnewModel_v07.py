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
            Demanda     = Mdemanda[i,j]
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

def ModeloRepoGRB(SKU ,Ts ,T ,P ,C ,F ,SCD0 ,I0 ,Me ,Tr ,B  ,Fvol ,semana):
    #combinaciones
    comb    = [(i,j,t) for i in SKU for j in Ts for t in T]
    combSCD = [(i,t) for i in SKU for t in T]
    #crear modelo
    model = Model('Repo')
    model.Params.LogToConsole = 0
    #agregar variables
    R   = model.addVars(comb ,vtype=GRB.INTEGER ,name='R')
    Q   = model.addVars(comb ,vtype=GRB.INTEGER ,name='Q')
    I   = model.addVars(comb ,vtype=GRB.INTEGER ,name='I')
    SCD = model.addVars(combSCD ,vtype=GRB.INTEGER ,name='SCD')
    opt = model.addVars(comb ,vtype=GRB.BINARY ,name='opt')
    #funcion objetivo
    m = 1
    n = 0
    z   = quicksum(Q[i,j,t] * (P[i,j][t] - C[i,j][t]) for i,j,t in comb) #ingreso economico por ventas
    c1  = quicksum(I[i,j,t] * Fvol[i] * np.log(t) for i,j,t in comb)         #costo por almacenar en tiendas
    c2  = quicksum(SCD[i,t] * Fvol[i] * np.log(t) for i,t in combSCD)        #costo por almacenar en CD
    c31 = quicksum(quicksum(I0[i,j] + R[i,j,semana] for i in SKU) * m / B[j][semana] + n for j in Ts)
    c32 = quicksum(quicksum(I[i,j,t-1] + R[i,j,t] for i in SKU) * m / B[j][t] + n for j in Ts for t in T if t!=semana)        #beneficio por mantener la tienda al tanto%
    c4 =  quicksum(opt[i,j,t] * 100000 for i,j,t in comb) #ingreso economico por ventas
    #model.setObjective(z  - c1- c2 + c31 + c32 ,GRB.MAXIMIZE)
    model.setObjective(z+c4,GRB.MAXIMIZE)

    #restricciones

    #reposicion no negativa
    model.addConstrs(R[i,j,t] >= 0 for i,j,t in comb)
    #cumplir minimos de exhibicion
    #model.addConstrs(I[i,j,t] >= Me[i,j][t] for i,j,t in comb)

    #reparticion no supera el stock disponible en CdD
    model.addConstrs(quicksum(R[i,j,t] for j in Ts for t in T) <= SCD[i,semana] for i in SKU)
    #no superar maximo almacenaje en tiendas
    model.addConstrs(quicksum(R[i,j,semana] + I0[i,j] for i in SKU) <= B[j][semana] for j in Ts)
    model.addConstrs(quicksum(R[i,j,t] + I[i,j,t-1] for i in SKU) <= B[j][t] for j in Ts for t in T if t!=semana)
    #continuidad de inventario
    model.addConstrs(R[i,j,semana] + I0[i,j] - Q[i,j,semana] - I[i,j,semana] == 0 for i in SKU for j in Ts)
    model.addConstrs(R[i,j,t] + I[i,j,t-1] - Q[i,j,t] - I[i,j,t] == 0 for i,j,t in comb if t!=semana)
    #continuidad de stock en CD
    model.addConstrs(SCD[i,semana] - SCD0[i] + quicksum(R[i,j,semana] for j in Ts) == 0 for i in SKU)
    model.addConstrs(SCD[i,t] - SCD[i,t-1] + quicksum(R[i,j,t] for j in Ts) == 0 for i,t in combSCD if t!=semana)

    #restricciones logicas
    model.addConstrs(Q[i,j,semana] <= I0[i,j] + R[i,j,semana] for i in SKU for j in Ts)
    model.addConstrs(Q[i,j,t] <= I[i,j,t-1] + R[i,j,t] for i,j,t in comb if t!=semana)
    model.addConstrs(Q[i,j,t] <= F[i,j][t] for i,j,t in comb)

    model.addConstrs((opt[i,j,semana] == 0) >> (Q[i,j,semana] >= I0[i,j] + R[i,j,semana]) for i in SKU for j in Ts)
    model.addConstrs((opt[i,j,t] == 0) >> (Q[i,j,t] >= I[i,j,t-1] + R[i,j,t]) for i,j,t in comb if t!=semana)
    model.addConstrs((opt[i,j,t] == 0) >> (Q[i,j,t] <= F[i,j][t] - 1) for i,j,t in comb)
    model.addConstrs((opt[i,j,t] == 1) >> (Q[i,j,t] >= F[i,j][t]) for i,j,t in comb)
    model.addConstrs((opt[i,j,t] == 1) >> (I[i,j,t] >= Me[i,j][t]) for i in SKU for j in Ts for t in T)

    #Restriccion limite de trasporte semanal
    model.addConstrs(quicksum(R[i,j,t] * Fvol[i] for i in SKU for j in Ts) <= Tr[t] for t in T)
    model.optimize()

    vals_repo  = { k : v.X for k,v in R.items() }
    vals_inve  = { k : v.X for k,v in I.items() }
    vals_venta = { k : v.X for k,v in Q.items() }
    vals_SCD   = { k : v.X for k,v in SCD.items() }
    vals_opt   = { k : v.X for k,v in opt.items() }
    return {'R':vals_repo , 'I' : vals_inve ,'Q' : vals_venta , 'SCD' : vals_SCD ,'opt' : vals_opt}

def ModeloVariasVentanas(Tt ,vT,SKU ,Ts ,P ,C ,F ,SCD0 ,I0 ,Me ,Tr ,B ,Fvol):
    #tiempo total en semanas
    TT = [ t+1 for t in range(Tt)]
    #Disc valores
    GRB_repo = {}
    GRB_inve = {}
    GRB_venta = {}
    GRB_SCD = {}
    GRB_opt = {}
    for sem in TT[:-(vT)]:
        T = np.arange(sem,sem+vT)
        print(T)
        vals    = ModeloRepoGRB(SKU ,Ts ,T ,P ,C ,F ,SCD0 ,I0 ,Me ,Tr ,B ,Fvol ,sem)
        #actualizo valores
        SCD0    = {i : vals['SCD'][i,sem] for i in SKU}
        I0      = {(i,j) : vals['I'][i,j,sem] for i in SKU for j in Ts}

        #guardo resultados(semana,i,j)
        GRB_repo[sem]   = {(i,j) : vals['R'][i,j,sem] for i in SKU for j in Ts}
        GRB_venta[sem]  = {(i,j) : vals['Q'][i,j,sem] for i in SKU for j in Ts}
        GRB_opt[sem]    = {(i,j) : vals['opt'][i,j,sem] for i in SKU for j in Ts}
        GRB_SCD[sem]    = SCD0
        GRB_inve[sem]   = I0
    return {'repo':GRB_repo ,'inventario': GRB_inve, 'SCD':  GRB_SCD, 'opt' :GRB_opt , 'venta': GRB_venta}

def obtenerCurvas(modelvals, SKU, Ts, Nsemanas):
    '''
    Recibe valores obtenidos del modelo en un diccionario con 5 keys(cada variable del modelo)
    cada una contiene un diccionario con claves desde el 1 a Nsemanas. cada semana tiene
    un tercer diccionario que contiene tiplas (i,j) con los SKU y tienda respectivamente como
    key y como valor el resultado de la optimizacion para esa variable.
    Retorna 3 matrices, la primera de 4 indices M[c,i,j,t]
        c: tipo de variable, 0:reposicion 1:inventario, 2: Venta/demanda 3:binario de quiebra de stock(0:quiebre 1:no quiebre)
        i: SKU
        j: tienda
        t: semana
    Segunda matriz R2 de el stock del CD StockCD[i,t]
    Tercera matriz de capacidad de la tienda en el tiempo CapT[j,t]
    '''
    NSKU    = len(SKU)
    NTs     = len(Ts)
    semanas = np.arange(1,Nsemanas + 1, dtype = np.int8)
    M       = np.zeros((4,NSKU,NTs,Nsemanas),dtype=np.int32)
    CapT    = np.zeros((NTs, Nsemanas),dtype=np.int32)
    StockCD = np.zeros((NSKU, Nsemanas),dtype=np.int32)

    for i in SKU:
        for j in Ts:
            for t in semanas:
                M[0,i-1,j-1,t-1] = modelvals['repo'][t][i,j]
                M[1,i-1,j-1,t-1] = modelvals['inventario'][t][i,j]
                M[2,i-1,j-1,t-1] = modelvals['venta'][t][i,j]
                M[3,i-1,j-1,t-1] = modelvals['opt'][t][i,j]
                if j==1:
                    StockCD[i-1,t-1] = modelvals['SCD'][t][i]



def plotQuiebres():
    pass

'''
*****************************************************
                    Parametros
*****************************************************
'''
t1 = time.time()
#x = np.arange(100)
#plt.plot(x,Normal(x,80,0.001,1))
#plt.show()
#precios
fracciones  = np.array([1.0 ,0.8 ,0.7 ,0.5 ,0.3])
semanas     = np.array([6 ,3 ,3 ,3 ,5])
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
#Mdemanda    = np.array([[300 ,300 ],
#                        [150 ,100]])
Mdemanda    = np.array([[500 ,50 ],
                        [15 ,100]])
Npeak       = 8
tamano      = 4

#trasporte
capacidad   = 1500
#capacidad   = np.sum(Mme) + 1500

#stock conjunto maximo en tiendas
Lcapacidad  = np.array([1000, 800])

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
#parametros temporales, ventana de tiempo
vT  = 8

Fvol = {i:1 for i in SKU}
#parametros auxiliares

'''
*****************************************************
                    Optimizacion
*****************************************************
'''

output_vars = ModeloVariasVentanas(Nsemanas ,vT,SKU ,Ts ,P ,C ,F ,SCD0 ,I0 ,Me ,Tr ,B ,Fvol)

'''
*****************************************************
                    Graficos
*****************************************************
'''


'''
*****************************************************
                    Fin
*****************************************************
'''
