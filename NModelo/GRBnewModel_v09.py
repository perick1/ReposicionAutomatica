import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

    Gz1 = 0
    Gz2 = 1

    Gb1 = 1
    Gc1 = 1

    z1 = quicksum(Q[i,j,t] * (P[i,j][t] - C[i,j][t]) * Gz1 * (T[-1] - t)**2 for i,j,t in comb)
    z2 = quicksum(Q[i,j,t] * Gz2 * (T[-1] - t)**2 for i,j,t in comb)

    B1 = quicksum(opt[i,j,t] * F[i,j][t] * Gb1 * (T[-1] - t)**2 for i,j,t in comb)
    C1 = quicksum(SCD[i,t] * Fvol[i] * Gc1 * t for i,t in combSCD)

    model.setObjective(z1 + z2 + B1 - C1 ,GRB.MAXIMIZE)

    #restricciones
    #reposicion no negativa
    model.addConstrs(R[i,j,t] >= 0 for i,j,t in comb)
    #reparticion no supera el stock disponible en CdD
    model.addConstrs(quicksum(R[i,j,semana] for j in Ts) <= SCD0[i] for i in SKU)
    model.addConstrs(quicksum(R[i,j,t] for j in Ts) <= SCD[i,t] for i,t in combSCD if t!=semana)
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
    model.addConstrs((opt[i,j,t] == 1) >> (Q[i,j,t] >= F[i,j][t]) for i,j,t in comb)

    model.addConstrs((opt[i,j,t] == 0) >> (Q[i,j,t] <= F[i,j][t] - 1) for i,j,t in comb)
    #model.addConstrs((opt[i,j,semana] == 1) >> (I0[i,j] >= Me[i,j][semana]) for i in SKU for j in Ts)
    model.addConstrs((opt[i,j,t] == 1) >> (I[i,j,t-1] >= Me[i,j][t]) for i,j,t in comb if t!=semana)

    #Restriccion limite de trasporte semanal
    model.addConstrs(quicksum(R[i,j,t] * Fvol[i] for i in SKU for j in Ts) <= Tr[t] for t in T)

    model.optimize()
    obj = model.getObjective()
    print(obj.getValue())

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
    for sem in TT[:-(vT)+1]:
        T = np.arange(sem,sem+vT)
        #print(T)
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
    for sem in T:
        #guardo los resiltados de la ultima iteracion tambien
        #print(sem)
        GRB_repo[sem]   = {(i,j) : vals['R'][i,j,sem] for i in SKU for j in Ts}
        GRB_venta[sem]  = {(i,j) : vals['Q'][i,j,sem] for i in SKU for j in Ts}
        GRB_opt[sem]    = {(i,j) : vals['opt'][i,j,sem] for i in SKU for j in Ts}
        GRB_SCD[sem]    = {i : vals['SCD'][i,sem] for i in SKU}
        GRB_inve[sem]   = {(i,j) : vals['I'][i,j,sem] for i in SKU for j in Ts}

    return {'repo':GRB_repo ,'inventario': GRB_inve, 'SCD':  GRB_SCD, 'opt' :GRB_opt , 'venta': GRB_venta}

def obtenerCurvas(modelvals, SKU, Ts, Nsemanas, SCD0, I0):
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
    #calculo ocupacion de las tiendas
    Inv0 = np.sum(I0,axis = 0).reshape((NTs,1))
    RepartidoT  = np.sum(M[0], axis = 0)
    InventarioT = np.sum(M[1], axis = 0)
    InventarioT = np.append(Inv0, InventarioT[:,:-1], axis = 1)
    CapT = RepartidoT + InventarioT
    #agrego stock inicial en cd
    Scd0 = np.array([SCD0[i] for i in SKU]).reshape((NSKU,1))
    StockCD = np.append(Scd0, StockCD, axis = 1)
    return [M ,StockCD,CapT]

def plotOcupaTienda(M,I0,Tcap,tienda):
    tienda  = tienda - 1
    repo    = M[0 ,: ,tienda ,:]
    inve    = M[1 ,: ,tienda ,:]
    Nsku, Nsemanas = repo.shape
    I0      = I0[:,tienda].reshape((Nsku,1))
    repo    = np.append(repo, np.zeros((Nsku,1)), axis = 1)
    inve    = np.append(I0, inve, axis = 1)
    #colores
    color = [c for name,c in mcolors.TABLEAU_COLORS.items()]
    #seteando parametros del grafico
    bar_width = 0.8
    y_offset = np.zeros(Nsemanas+1)
    x = np.arange(Nsemanas+1)
    fig, ax = plt.subplots(figsize = (6,2.5))
    #ploteo barras
    for i in range(Nsku):
        print(inve[i])
        y =  (repo[i] + inve[i]) / Tcap[tienda]
        ax.bar(x, y, bar_width, bottom=y_offset,alpha = 0.5, color=color[i],label = f'SKU {i+1}')
        y_offset = y_offset + y
    ax.set_title(f'Ocupacion de tienda {tienda+1} por semana, normalizada.')
    ax.set_xlim(-0.5,Nsemanas+0.5)
    ax.set_ylim(0,1.1)
    ax.xaxis.set_ticks([n for n in range(Nsemanas+1)])
    ax.legend()
    ax.grid(axis = 'both' ,color='gray', linestyle='--', linewidth=0.5)
    plt.show(block=False)
    fig.tight_layout()

def plotSCDhistorico(scd_model,SCD0):
    Nsku, Nsemanas = scd_model.shape
    #colores
    color = [c for name,c in mcolors.TABLEAU_COLORS.items()]
    #seteando parametros del grafico
    bar_width = 0.8
    y_offset = np.zeros(Nsemanas)
    x = np.arange(Nsemanas)

    fig, ax = plt.subplots(figsize = (6,2.5))
    for i in range(Nsku):
        y =  (scd_model[i]) / SCD0[i+1]
        ax.bar(x, y, bar_width, bottom=y_offset,alpha = 0.5, color=color[i],label = f'SKU {i+1}')
        y_offset = y_offset + y
    ax.set_title(f'Stock en Centro de Distribucion, normalizado.')
    ax.set_xlim(-0.5,Nsemanas-0.5)
    ax.set_ylim(0,Nsku + 0.2)
    ax.xaxis.set_ticks([n for n in range(Nsemanas)])
    ax.legend()
    ax.grid(axis = 'both' ,color='gray', linestyle='--', linewidth=0.5)
    plt.show(block=False)
    fig.tight_layout()

def plotRepoTransporte(Mrepo, CapTr):
    Nsku, Ntiendas, Nsemanas = Mrepo.shape
    #colores
    color = [c for name,c in mcolors.TABLEAU_COLORS.items()]
    color = color[:Nsku*Ntiendas]
    color = np.reshape(color, (Nsku,Ntiendas))
    #parametros del grafico
    bar_width = 0.8
    x = np.arange(1,Nsemanas+1)
    y_offset = np.zeros(Nsemanas)
    fig, ax = plt.subplots(figsize = (6,2.5))

    ax.fill_between([0,Nsemanas+1],0,1,facecolor='green', alpha=0.3)
    for i in range(Nsku):
        for j in range(Ntiendas):
            y = Mrepo[i,j] / CapTr
            ax.bar(x, y, bar_width, bottom=y_offset,alpha = 0.5, color=color[i,j],label = f'SKU {i+1} en tienda {j+1}')
            y_offset = y_offset + y

    ax.set_title('Reposiciones por semana, normalizadas a la capacidad de transporte')
    ax.set_ylim(0,1.1)
    ax.set_xlim(0.5, Nsemanas + 0.5)
    ax.xaxis.set_ticks([n+1 for n in range(Nsemanas)])
    ax.legend()
    ax.grid(axis = 'both' ,color='gray', linestyle='--', linewidth=0.5)
    plt.show(block=False)
    fig.tight_layout()

def plotRepoQuiebres(M, I0, F,vT, tienda, sku):
    tienda  = tienda - 1
    sku     = sku - 1
    repo    = M[0 ,sku,tienda]
    inve    = M[1 ,sku,tienda]
    Q       = M[2 ,sku,tienda]
    opt     = M[3 ,sku,tienda]
    Nsemanas= len(repo)
    I0      = I0[sku,tienda]
    inve    = np.append([I0], inve[:-1])
    #grafico
    fig, ax = plt.subplots(figsize = (6.5,3))
    x_offset= 0.24
    x       = np.arange(1,Nsemanas+1)
    xfill     = np.copy(x)
    xfill[0]  = x[0] - 1
    xfill[-1] = x[-1] + 1
    bar_width = 0.45
    D = [F[sku +1, tienda +1][t] for t in range(1,Nsemanas+1)]
    #D = F[sku + 1, tienda + 1].values()
    max   = np.max([np.max(repo+inve),np.max(D)])

    #r_bar_he = np.mean([maxinv,np.mean(D)])
    r_bar_he = 0.45
    x_rbar   = np.arange(Nsemanas-vT,Nsemanas+1) +0.5
    print(inve)

    ax.fill_between(xfill,D/max,1.2,facecolor='g', alpha=0.4,label = 'Sobredemanda')
    ax.fill_between(x_rbar,0.45,0.55,facecolor='red', alpha=0.6,label = 'Ventana opt.')
    ax.bar(x - x_offset, inve/max, bar_width, color='gray',alpha = 0.5,label = 'Inventario')
    ax.bar(x - x_offset, repo/max, bar_width, bottom=inve, color='blue',alpha = 0.5,label = 'Reposicion')
    ax.bar(x + x_offset, Q/max, bar_width, color='orange',alpha = 0.5,label = 'Demanda ajustada')
    print(opt)

    for t in range(Nsemanas):
        if opt[t] == 0:
            ytext = ((inve[t]+repo[t])/max)
            xtext = x[t] - 0.25
            print(xtext,ytext)
            ax.annotate('x',xy=(xtext, ytext), xycoords='data',fontsize=15,color = 'red',weight = 'bold')

    ax.set_title(f'Optimizacion 20/8 semanas.SKU {sku + 1}, tienda {tienda+1}. Normalizado')
    ax.set_xlim(0.5,Nsemanas+0.5)
    ax.set_ylim(0,1.1)
    ax.xaxis.set_ticks([n for n in range(1,Nsemanas+1)])
    ax.legend(ncol = 3,loc='center left', bbox_to_anchor=(0.05, -0.3))
    ax.grid(axis = 'both' ,color='gray', linestyle='--', linewidth=0.5)
    plt.show(block=False)
    fig.tight_layout()

'''
*****************************************************
                    Parametros
*****************************************************
'''
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
capacidad   = 2000
#capacidad   = np.sum(Mme) + 1500

#stock conjunto maximo en tiendas
Lcapacidad  = np.array([1000, 800])

#inventario inicial periodo cero
Minv        = np.array([[0 ,0 ],
                        [0 ,0]])
#Minv        = Mme

#stock en centro de distribucion
SCD0 =  {1:1000,
         2:12000}

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
t1 = time.time()
output_vals = ModeloVariasVentanas(Nsemanas ,vT,SKU ,Ts ,P ,C ,F ,SCD0 ,I0 ,Me ,Tr ,B ,Fvol)
Mvals, scd_model, ocupado_tiendas = obtenerCurvas(output_vals, SKU, Ts, Nsemanas, SCD0, Minv)
RepartidoT = Mvals[0]
Transportado = np.sum(np.sum(RepartidoT,axis = 0),axis=0)
t2 = time.time()
print('tiempo de ejecucion: ',round(t2-t1,2))
'''
*****************************************************
                    Graficos
*****************************************************
'''
tienda = 1
plotOcupaTienda(Mvals,Minv,Lcapacidad,tienda)
tienda = 2
plotOcupaTienda(Mvals,Minv,Lcapacidad,tienda)
plotSCDhistorico(scd_model,SCD0)
plotRepoTransporte(Mvals[0], capacidad)
sku = 2
plotRepoQuiebres(Mvals, Minv, F, vT, tienda, sku)
'''
*****************************************************
                    Fin
*****************************************************
'''
