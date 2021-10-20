import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

'''
*****************************************************
                    Parametros
*****************************************************
'''
TT = list(range(1,14))

#stock en centro de distribucion
SCD =  {1:100000,
        2:120000}
#inventario inicial periodo cero
I0  =  {(1,1):0,
        (1,2):0,
        (2,1):0,
        (2,2):0}

for semana in TT[:-3]:
    T   = list(range(semana,semana+4))
    SKU = [1 ,2]
    Ts  = [1 ,2]
    #factores
    Fvol = {i:1 for i in SKU}
    F1   = {t:2*t for t in TT}
    F2   = {t:3*t for t in TT}
    Fd   = 2
    #precios
    P   =  {(1,1):{1:200*4 ,2:200*4 ,3:200*4 ,4:200*4, 5:200*4 ,6:200*4 ,7:200*4*0.8 ,8:200*4*0.8 ,9:200*4*0.7 ,10:200*4*0.7 ,11:200*4*0.5 ,12:200*4*0.5,13:200*4*0.3},
            (1,2):{1:300*4 ,2:300*4 ,3:300*4 ,4:300*4, 5:300*4 ,6:300*4 ,7:300*4*0.8 ,8:300*4*0.8 ,9:300*4*0.7 ,10:300*4*0.7 ,11:300*4*0.5 ,12:300*4*0.5,13:300*4*0.3},
            (2,1):{1:400*4 ,2:400*4 ,3:400*4 ,4:400*4, 5:400*4 ,6:400*4 ,7:400*4*0.8 ,8:400*4*0.8 ,9:400*4*0.7 ,10:400*4*0.7 ,11:400*4*0.5 ,12:400*4*0.5,13:400*4*0.3},
            (2,2):{1:100*4 ,2:100*4 ,3:100*4 ,4:100*4, 5:100*4 ,6:100*4 ,7:100*4*0.8 ,8:100*4*0.8 ,9:100*4*0.7 ,10:100*4*0.7 ,11:100*4*0.5 ,12:100*4*0.5,13:100*4*0.3}}
    #costos
    C   =  {(1,1):{1:200 ,2:200 ,3:200 ,4:200, 5:200 ,6:200 ,7:200 ,8:200, 9:200 ,10:200 ,11:200 ,12:200, 13:200},
            (1,2):{1:300 ,2:300 ,3:300 ,4:300, 5:300 ,6:300 ,7:300 ,8:300, 9:300 ,10:300 ,11:300 ,12:300, 13:300},
            (2,1):{1:400 ,2:400 ,3:400 ,4:400, 5:400 ,6:400 ,7:400 ,8:400, 9:400 ,10:400 ,11:400 ,12:400, 13:400},
            (2,2):{1:100 ,2:100 ,3:100 ,4:100, 5:100 ,6:100 ,7:100 ,8:100, 9:100 ,10:100 ,11:100 ,12:100, 13:100}}
    #IVA
    IVA =  {(1,1):{1:200*0.19 ,2:200*0.19 ,3:200*0.19 ,4:200*0.19, 5:200*0.19 ,6:200*0.19 ,7:200*0.19 ,8:200*0.19, 9:200*0.19 ,10:200*0.19 ,11:200*0.19 ,12:200*0.19, 13:200*0.19},
            (1,2):{1:300*0.19 ,2:300*0.19 ,3:300*0.19 ,4:300*0.19, 5:300*0.19 ,6:300*0.19 ,7:300*0.19 ,8:300*0.19, 9:300*0.19 ,10:300*0.19 ,11:300*0.19 ,12:300*0.19, 13:300*0.19},
            (2,1):{1:400*0.19 ,2:400*0.19 ,3:400*0.19 ,4:400*0.19, 5:400*0.19 ,6:400*0.19 ,7:400*0.19 ,8:400*0.19, 9:400*0.19 ,10:400*0.19 ,11:400*0.19 ,12:400*0.19, 13:400*0.19},
            (2,2):{1:100*0.19 ,2:100*0.19 ,3:100*0.19 ,4:100*0.19, 5:100*0.19 ,6:100*0.19 ,7:100*0.19 ,8:100*0.19, 9:100*0.19 ,10:100*0.19 ,11:100*0.19 ,12:100*0.19, 13:100*0.19}}
    #Neto
    N   = C.copy()
    for i in SKU:
        for j in Ts:
            for t in T:
                N[(i,j)][t] = P[(i,j)][t] - C[(i,j)][t] - IVA[(i,j)][t]

    #minimos de exhibicion
    Me  =  {(1,1):{1:50 ,2:50 ,3:50 ,4:50, 5:50 ,6:50 ,7:50 ,8:50, 9:50 ,10:50 ,11:50 ,12:50, 13:50},
            (1,2):{1:30 ,2:30 ,3:30 ,4:30, 5:30 ,6:30 ,7:30 ,8:30, 9:30 ,10:30 ,11:30 ,12:30, 13:30},
            (2,1):{1:80 ,2:80 ,3:80 ,4:80, 5:80 ,6:80 ,7:80 ,8:80, 9:80 ,10:80 ,11:80 ,12:80, 13:80},
            (2,2):{1:120 ,2:120 ,3:120 ,4:120, 5:120 ,6:120 ,7:120 ,8:120, 9:120 ,10:120 ,11:120 ,12:120, 13:120}}
    #stock conjunto en tienda
    B   =  {1:{1:10000 ,2:10000 ,3:10000 ,4:10000, 5:10000 ,6:10000 ,7:10000, 8:10000 ,9:10000 ,10:10000, 11:10000 ,12:10000 ,13:10000},
            2:{1:15000 ,2:15000 ,3:15000 ,4:15000, 5:15000 ,6:15000 ,7:15000, 8:15000 ,9:15000 ,10:15000, 11:15000 ,12:15000 ,13:15000}}
    #demanda
    D   =  {(1,1):{1:400 ,2:400 ,3:400 ,4:400*2 ,5:400*4 ,6:400*2 ,7:400, 8:400 ,9:400 ,10:400, 11:400 ,12:400 ,13:400},
            (1,2):{1:500 ,2:500 ,3:500 ,4:500*2 ,5:500*4 ,6:500*2 ,7:500, 8:500 ,9:500 ,10:500, 11:500 ,12:500 ,13:500},
            (2,1):{1:200 ,2:200 ,3:200 ,4:200*2 ,5:200*4 ,6:200*2 ,7:200, 8:200 ,9:200 ,10:200, 11:200 ,12:200 ,13:200},
            (2,2):{1:600 ,2:600 ,3:600 ,4:600*2 ,5:600*4 ,6:600*2 ,7:600, 8:600 ,9:600 ,10:600, 11:600 ,12:600 ,13:600}}
    #capacidad de transporte
    Tr = {1:5000 ,2:5000 ,3:5000 ,4:5000 ,5:5000 ,6:5000 ,7:5000, 8:5000 ,9:5000 ,10:5000, 11:5000 ,12:5000 ,13:5000}

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
    z  = quicksum(V[i,j,t] * N[i,j][t] for i in SKU for j in Ts for t in T)
    c1 = quicksum(I[i,j,t] * Fvol[i] * F1[t] for i in SKU for j in Ts for t in T)
    c2 = quicksum((SCD[i] -  quicksum(R[i,j,tau] for j in Ts for tau in T[:t])) * Fvol[i] * F2[t] for i in SKU for t in T)
    model.setObjective(z - c1 - c2 ,GRB.MAXIMIZE)
    #model.setObjective(z -c2,GRB.MAXIMIZE)

    '''
    *****************************************************
                        Restricciones
    *****************************************************
    '''
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

    '''
    *****************************************************
                        Resultados
    *****************************************************
    '''
    #model.display()
    model.optimize()
    print('Funcion Objetivo: ',str(round(model.ObjVal,2)))
    #for v in model.getVars():
    #    print(str(v.VarName)+' = '+str(round(v.x,2)))
    '''
    *****************************************************
                        Graficos
    *****************************************************
    '''
    if semana == 2:
        fig, axs = plt.subplots(nrows=len(SKU) ,ncols=len(Ts) ,figsize = (13,8))
    repartido = [0,0]
    inv_next = [[0,0],
                [0,0]]
    for i in SKU:
        for j in Ts:

            sku = i
            tienda = j

            r    = [R[sku,tienda,t].x for t in T] + [0]
            repartido[i-1] += sum(r)
            #print(sum(r))
            v    = [V[sku,tienda,t].x for t in T]
            inv  = [I[sku,tienda,t].x for t in T]
            inv_next[i-1][j-1] = inv[0]
            inv  = [I0[sku,tienda]] + inv
            d    = [D[sku,tienda][t] for t in T]
            T2   = T + [5]
            if semana == 2:
                y_offset = np.zeros(5)
                bar_width = 0.6

                ax = axs[i-1,j-1]
                ax.plot(T,d,ls = '-',color = 'red',label = 'demanda')
                ax.plot(T,v,ls = '',marker = '*',color = 'blue',label = 'ventas')
                ax.bar(T2, inv, bar_width, bottom=y_offset, color='gray',label = 'inventario')
                y_offset = y_offset + np.array(inv)
                ax.bar(T2, r, bar_width, bottom=y_offset, color='green',label = 'reposicion')
                ax.set_title(f'SKU {sku} en tienda {tienda}')
                ax.legend()
                plt.show(block=False)
                fig.tight_layout()
        #print(repartido[i])

    #stock en centro de distribucion
    SCD =  {1:SCD[1] - repartido[0],
            2:SCD[2] - repartido[1]}
    #inventario inicial periodo cero
    I0  =  {(1,1):inv_next[0][0],
            (1,2):inv_next[0][1],
            (2,1):inv_next[1][0],
            (2,2):inv_next[1][1]}
    '''
    *****************************************************

    *****************************************************
    '''
