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

    Gz1 = 1 #precio
    Gz2 = 1 #solo cantidad

    Gb1 = 1 #beneficio por evitar quiebres
    Gc1 = 1 #costo por tardar en vender

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
    model.addConstrs((opt[i,j,t] == 0) >> (Q[i,j,t] <= F[i,j][t] - 1) for i,j,t in comb)
    model.addConstrs((opt[i,j,t] == 1) >> (Q[i,j,t] >= F[i,j][t]) for i,j,t in comb)
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

def ModeloRepoGRB1semana(SKU ,Ts ,P ,C ,F ,SCD0 ,I0 ,Me ,Tr ,B  ,Fvol ,semana):
    t = semana
    #combinaciones
    comb    = [(i,j) for i in SKU for j in Ts]
    #crear modelo
    model = Model('Repo')
    model.Params.LogToConsole = 0
    #agregar variables
    R   = model.addVars(comb ,vtype=GRB.INTEGER ,name='R')
    Q   = model.addVars(comb ,vtype=GRB.INTEGER ,name='Q')
    I   = model.addVars(comb ,vtype=GRB.INTEGER ,name='I')
    SCD = model.addVars(SKU ,vtype=GRB.INTEGER ,name='SCD')
    opt = model.addVars(comb ,vtype=GRB.BINARY ,name='opt')
    #funcion objetivo

    Gz1 = 0 #precio
    Gz2 = 1 #solo cantidad

    Gb1 = 1 #beneficio por evitar quiebres
    Gc1 = 1 #costo por tardar en vender

    z1 = quicksum(Q[i,j] * (P[i,j][t] - C[i,j][t]) * Gz1  for i,j in comb)
    z2 = quicksum(Q[i,j] * Gz2  for i,j in comb)

    B1 = quicksum(opt[i,j] * F[i,j][t] * Gb1  for i,j in comb)
    C1 = quicksum(SCD[i] * Fvol[i] * Gc1  for i in SKU)

    model.setObjective(z1 + z2 + B1 - C1 ,GRB.MAXIMIZE)

    #restricciones
    #reposicion no negativa
    model.addConstrs(R[i,j] >= 0 for i,j in comb)
    #reparticion no supera el stock disponible en CdD
    model.addConstrs(quicksum(R[i,j] for j in Ts) <= SCD0[i] for i in SKU)
    #no superar maximo almacenaje en tiendas
    model.addConstrs(quicksum(R[i,j] + I0[i,j] for i in SKU) <= B[j][semana] for j in Ts)
    #continuidad de inventario
    model.addConstrs(R[i,j] + I0[i,j] - Q[i,j] - I[i,j] == 0 for i,j in comb)
    #continuidad de stock en CD
    model.addConstrs(SCD[i] - SCD0[i] + quicksum(R[i,j] for j in Ts) == 0 for i in SKU)
    #restricciones logicas
    model.addConstrs(Q[i,j] <= I0[i,j] + R[i,j] for i,j in comb)
    model.addConstrs(Q[i,j] <= F[i,j][t] for i,j in comb)

    model.addConstrs((opt[i,j] == 0) >> (Q[i,j] >= I0[i,j] + R[i,j]) for i,j in comb)
    model.addConstrs((opt[i,j] == 0) >> (Q[i,j] <= (F[i,j][t] - 1)) for i,j in comb)
    model.addConstrs((opt[i,j] == 1) >> (Q[i,j] >= F[i,j][t]) for i,j in comb)
    model.addConstrs((opt[i,j] == 1) >> (I[i,j] >= Me[i,j][t]) for i,j in comb)

    #Restriccion limite de trasporte semanal
    model.addConstr(quicksum(R[i,j] * Fvol[i] for i in SKU for j in Ts) <= Tr[t])

    model.optimize()
    #if semana == 5:
    #    print(model.display())
    obj = model.getObjective()
    print(obj.getValue())

    vals_repo  = { k : v.X for k,v in R.items() }
    vals_inve  = { k : v.X for k,v in I.items() }
    vals_venta = { k : v.X for k,v in Q.items() }
    vals_SCD   = { k : v.X for k,v in SCD.items() }
    vals_opt   = { k : v.X for k,v in opt.items() }
    return {'R':vals_repo , 'I' : vals_inve ,'Q' : vals_venta , 'SCD' : vals_SCD ,'opt' : vals_opt}

def ModeloVariasVentanas1semana(Tt ,SKU ,Ts ,P ,C ,F ,SCD0 ,I0 ,Me ,Tr ,B ,Fvol):
    #tiempo total en semanas
    TT = [ t+1 for t in range(Tt)]
    #Disc valores
    GRB_repo = {}
    GRB_inve = {}
    GRB_venta = {}
    GRB_SCD = {}
    GRB_opt = {}
    for sem in TT:
        #print(T)
        vals    = ModeloRepoGRB1semana(SKU ,Ts ,P ,C ,F ,SCD0 ,I0 ,Me ,Tr ,B ,Fvol ,sem)
        #actualizo valores
        SCD0    = {i : vals['SCD'][i] for i in SKU}
        I0      = {(i,j) : vals['I'][i,j] for i in SKU for j in Ts}

        #guardo resultados(semana,i,j)
        GRB_repo[sem]   = {(i,j) : vals['R'][i,j] for i in SKU for j in Ts}
        GRB_venta[sem]  = {(i,j) : vals['Q'][i,j] for i in SKU for j in Ts}
        GRB_opt[sem]    = {(i,j) : vals['opt'][i,j] for i in SKU for j in Ts}
        GRB_SCD[sem]    = SCD0
        GRB_inve[sem]   = I0

        dic_repo1 = {'repo':GRB_repo ,'inventario': GRB_inve, 'SCD':  GRB_SCD, 'opt' :GRB_opt , 'venta': GRB_venta}
        #print(dic_repo1)

    return dic_repo1

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
    fig, ax = plt.subplots(figsize = (6,3))
    #ploteo barras
    for i in range(Nsku):
        #print(inve[i])
        y =  (repo[i] + inve[i]) / Tcap[tienda]
        ax.bar(x, y, bar_width, bottom=y_offset,lw = 1.3,edgecolor='k',alpha = 0.5, color=color[i],label = f'SKU 0{i+1}')
        y_offset = y_offset + y
    ax.set_title(f'Ocupacion de tienda {tienda+1} por semana')
    ax.set_xlim(-0.5,Nsemanas+0.5)
    ax.set_ylim(0,1.1)
    ax.set_xlabel('Semana')
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

    fig, ax = plt.subplots(figsize = (6,3))
    for i in range(Nsku):
        y =  (scd_model[i]) / SCD0[i+1]
        ax.bar(x, y, bar_width, bottom=y_offset,lw = 1.3,edgecolor='k',alpha = 0.5, color=color[i],label = f'SKU 0{i+1}')
        y_offset = y_offset + y
    ax.set_title(f'Stock en Centro de Distribucion')
    ax.set_xlim(-0.5,Nsemanas-0.5)
    ax.set_xlabel('Semana')
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
    fig, ax = plt.subplots(figsize = (7.5,4))

    ax.fill_between([0,Nsemanas+1],0,1,facecolor='green', alpha=0.3, label = 'Capacidad de transporte')
    for i in range(Nsku):
        for j in range(Ntiendas):
            y = Mrepo[i,j] / CapTr
            ax.bar(x, y, bar_width, bottom=y_offset,lw = 1.3,edgecolor='k',alpha = 0.5, color=color[i,j],label = f'SKU 0{i+1} en tienda 0{j+1}')
            y_offset = y_offset + y

    ax.set_title('Reposiciones por semana')
    ax.set_ylim(0,1.1)
    ax.set_xlim(0.5, Nsemanas + 0.5)
    ax.xaxis.set_ticks([n+1 for n in range(Nsemanas)])
    #ax.legend()
    ax.legend(ncol = 3,loc='center left', bbox_to_anchor=(-0.01, -0.4))
    ax.grid(axis = 'both' ,color='gray', linestyle='--', linewidth=0.5)
    plt.show(block=False)
    fig.tight_layout()

def plotRepoQuiebres(M, I0, F, RF, vT, tienda, sku,std):
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
    fig, ax = plt.subplots(figsize = (4.5,3))
    x_offset= 0.24
    x       = np.arange(1,Nsemanas+1)
    xfill     = np.copy(x)
    xfill[0]  = x[0] - 1
    xfill[-1] = x[-1] + 1
    bar_width = 0.45
    D = [F[sku +1, tienda +1][t] for t in range(1,Nsemanas+1)]
    RD= [RF[sku +1, tienda +1][t] for t in range(1,Nsemanas+1)]
    #D = F[sku + 1, tienda + 1].values()
    max   = np.max([np.max(repo+inve),np.max(D)])
    #max   = max * 0 + 1

    #r_bar_he = np.mean([maxinv,np.mean(D)])
    r_bar_he = 0.45
    x_rbar   = np.arange(Nsemanas-vT,Nsemanas+1) +0.5
    #print(inve)

    ax.fill_between(xfill,D/max,1.5,facecolor='g', alpha=0.3,label = 'Sobredemanda')
    ax.fill_between(x_rbar,0.45,0.55,facecolor='red', alpha=0.6,label = 'Ventana opt.')
    ax.bar(x - x_offset, inve/max, bar_width, lw = 1.3,edgecolor='k',color='gray',alpha = 0.5,label = 'Inventario')
    ax.bar(x - x_offset, repo/max, bar_width, lw = 1.3,edgecolor='k',bottom=inve/max, color='blue',alpha = 0.5,label = 'Reposicion')
    ax.bar(x + x_offset, Q/max, bar_width, lw = 1.3,edgecolor='k',color='orange',alpha = 0.5,label = 'Demanda ajustada')
    ax.plot(x,RD/max,color='darkgreen', lw = 2,alpha=0.8,label = f'Forecast Ruido {std}')
    #print(opt)

    for t in range(Nsemanas):
        if opt[t] == 0:
            ytext = ((inve[t]+repo[t])/max)
            xtext = x[t] - 0.25
            #print(xtext,ytext)
            ax.annotate('x',xy=(xtext, ytext), xycoords='data',fontsize=15,color = 'red',weight = 'bold')

    ax.set_title(f'SKU: 0{sku + 1} Tienda: 0{tienda+1}.')
    ax.set_xlim(0.5,Nsemanas+0.5)
    ax.set_xlabel('Semana')
    ax.set_ylim(0,1.3)
    ax.xaxis.set_ticks([n for n in range(1,Nsemanas+1)])
    #ax.legend(ncol = 3,loc='center left', bbox_to_anchor=(0.01, -0.3))
    #ax.legend(ncol = 1,loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis = 'both' ,color='gray', linestyle='--', linewidth=0.5)
    plt.show(block=False)
    fig.tight_layout()

def Ganancias(Q, P, C, SKU, Ts, Nsemanas):
    T = [i  for i in range(1,Nsemanas+1)]
    comb    = [(i,j) for i in SKU for j in Ts]
    Ventas = Q[:,:,:Nsemanas]
    print(Ventas.shape)
    GSemanal = np.zeros(Nsemanas)
    CSemanal = np.zeros(Nsemanas)
    for t in T:
        G = [(Q[i-1,j-1,t-1] * (P[i,j][t] - C[i,j][t])) for i,j in comb]
        CS= [Q[i-1,j-1,t-1]  for i,j in comb]
        GSemanal[t-1] = np.sum(G)
        CSemanal[t-1] = np.sum(CS)
    return [GSemanal.astype(np.int32),CSemanal.astype(np.int32)]

def plotResumen4(M, I0, F,Rf,vT, SKU, Ts,std=0.0):
    for i in SKU:
        for j in Ts:
            plotRepoQuiebres(Mvals, Minv, F,RF, vT, j, i,std)

def RuidoGRB(Tt ,vT,SKU ,Ts ,P ,C ,F ,SCD0 ,I0 ,Me ,Tr ,B ,Fvol, std):
    '''
    Cuento los quiebres de stock producidos por agregar ruido a la Demanda
    tambie la ganancia en ventas
    '''
    np.random.seed(32)
    iAux    = I0.copy()
    #tiempo total en semanas
    TT = [ t+1 for t in range(Tt)]

    GRB_repo = {}
    GRB_inve = {}
    GRB_venta = {}
    GRB_SCD = {}
    GRB_opt = {}
    Mruido  = np.random.normal(0, std, size = (len(SKU),len(Ts),Tt - vT + 1))
    #print(Mruido)

    for sem in TT[:-(vT)+1]:
        T       = np.arange(sem,sem+vT)
        print(T)
        #scdAux  = SCD0.copy()
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

        #corrijo por error en forecast (ruido)
        for i in SKU:
            for j in Ts:
                RuidoF = round(Mruido[i-1,j-1,sem-1] * F[i,j][1])
                #ruido no puede superar el forecast negativamente
                RuidoF = max(2 - F[i,j][sem], RuidoF)
                if sem == 1:
                    errRepo = abs(iAux[i,j] + GRB_repo[sem][i,j] - F[i,j][sem])
                else:
                    errRepo = abs(GRB_inve[sem-1][i,j] + GRB_repo[sem][i,j] - F[i,j][sem])
                #ya habia cumplido la demanda
                if GRB_opt[sem][i,j] == 1:
                    if RuidoF > errRepo:
                        #print(f'{GRB_inve[sem-1][i,j] + GRB_repo[sem][i,j]},{F[i,j][sem]},H')
                        #print(f'{i},{j},{sem},H')
                        GRB_opt[sem][i,j]   = 0
                        GRB_venta[sem][i,j] = GRB_inve[sem-1][i,j] + GRB_repo[sem][i,j]
                        GRB_inve[sem][i,j]  = 0
                    elif RuidoF > 0:
                        #print(f'{i},{j},{sem},O')
                        GRB_venta[sem][i,j] = F[i,j][sem] + RuidoF
                        GRB_inve[sem][i,j]  = GRB_inve[sem][i,j] - RuidoF
                        GRB_inve[sem][i,j]  = errRepo - RuidoF
                    #elif RuidoF < 0:
                    else:
                        #print(f'{i},{j},{sem},L')
                        #print(f'{GRB_venta[sem][i,j]},{RuidoF},H')
                        GRB_venta[sem][i,j] = GRB_venta[sem][i,j] + RuidoF
                        GRB_inve[sem][i,j]  = GRB_inve[sem][i,j] - RuidoF
                        #print(f'{GRB_venta[sem][i,j]},{RuidoF},H')
                elif RuidoF < (-errRepo):
                    Qnuevo = F[i,j][sem] + RuidoF
                    GRB_opt[sem][i,j]   = 1
                    GRB_inve[sem][i,j]  = GRB_venta[sem][i,j] - Qnuevo
                    GRB_venta[sem][i,j] = Qnuevo
                    #print(f'{i},{j},{sem},A')
                #else:
                    #print(GRB_opt[sem][i,j])
                    #print(f'{i},{j},{sem},S')
                    #print(GRB_opt[sem][i,j])
                Mruido[i-1,j-1,sem-1] = RuidoF

    for sem in T[1:]:
        GRB_repo[sem]   = {(i,j) : vals['R'][i,j,sem] for i in SKU for j in Ts}
        GRB_venta[sem]  = {(i,j) : vals['Q'][i,j,sem] for i in SKU for j in Ts}
        GRB_opt[sem]    = {(i,j) : vals['opt'][i,j,sem] for i in SKU for j in Ts}
        GRB_SCD[sem]    = {i : vals['SCD'][i,sem] for i in SKU}
        GRB_inve[sem]   = {(i,j) : vals['I'][i,j,sem] for i in SKU for j in Ts}
    return {'repo':GRB_repo ,'inventario': GRB_inve, 'SCD':  GRB_SCD, 'opt' :GRB_opt , 'venta': GRB_venta, 'Ruido': Mruido}

def curvaForcastRuido(MR,F,SKU,Ts):
    r,c,Ttotal = MR.shape
    Ndict = {}
    for i in SKU:
        for j in Ts:
            N = F[i,j].copy()
            for t in range(1,Ttotal + 1):
                N[t] = F[i,j][t] + MR[i-1,j-1,t-1]
            Ndict[i,j] = N
    return Ndict

def TodaslasCurvas(Mvals, Minv, F,RF,vT, SKU, Ts, STD,capacidad,scd_model,SCD0,Lcapacidad):
    fig, axs = plt.subplots(nrows = 4, ncols = 2,figsize = (9.2,9.5))
    st_color = [['orangered','orange'],
                ['blue','teal']]
    f_color  = ['green','black']

    s_color   = ['lightcoral','deepskyblue']
    extra_color = ['purple','black','seagreen','lightcoral','red','c','g','pink','slategrey','mintcream']
    #axs = axs.flat
    #print(axs)
    Nsemanas= len(Mvals[0,0,0])
    y_offset_tr = np.zeros(Nsemanas)

    for tienda in Ts:
        tienda_1    = tienda - 1
        axt         = axs[2,tienda_1]
        y_offset    = np.zeros(Nsemanas)
        y_offset_scd= np.zeros(Nsemanas)

        for sku in SKU:
            sku_1     = sku - 1

            Nsemanas= len(Mvals[0,0,0])

            ax      = axs[sku_1,tienda_1]
            repo    = Mvals[0 ,sku_1,tienda_1]
            inve    = Mvals[1 ,sku_1,tienda_1]
            #Q       = M[2 ,sku,tienda]
            opt     = Mvals[3 ,sku_1,tienda_1]
            I0      = Minv[sku_1,tienda_1]
            inve    = np.append([I0], inve[:-1])
            D       = [ F[sku, tienda][t] for t in range(1,Nsemanas+1)]
            RD      = [RF[sku, tienda][t] for t in range(1,Nsemanas+1)]

            x           = np.arange(1,Nsemanas+1)
            bar_width   = 0.9
            x_rbar      = np.arange(Nsemanas-vT,Nsemanas+1) +0.5
            x_rbar      = np.arange(0,Nsemanas+2)
            Xmarker     = repo + inve
            Xzero       = opt==1
            Xmarker[Xzero] = np.zeros(np.sum(Xzero)) - 200000

            max = np.max(D)
            ax.fill_between(x_rbar,0.0,2.0,facecolor=extra_color[-1], alpha=0.5)#,label = 'Ventana última iteración')

            ax.bar(x, inve/max, bar_width,lw = 3,edgecolor='white',color=st_color[sku_1][tienda_1],alpha = 0.4)#,label = 'Inventario')
            ax.bar(x, repo/max, bar_width,lw = 3,edgecolor='white',color=st_color[sku_1][tienda_1],alpha = 0.8,bottom=inve/max,label = f'Stock SKU:0{sku} Tienda:0{tienda}')

            text = [['(a)','(b)'],
                    ['(c)','(d)'],
                    ['(e)','(f)'],
                    ['(g)','(h)']]

            ax.annotate(text[sku_1][tienda_1],xy=(0.93,1.05), xycoords='axes fraction',fontsize=12,color = 'black')#,weight = 'bold')

            ax.set_title(f'Stock SKU: 0{sku} Tienda: 0{tienda}',fontsize = 10)
            ax.set_xlim(0.5,Nsemanas+0.5)
            axt.set_xlim(0.5,Nsemanas+0.5)
            ax.set_xlabel('Semana',fontsize = 9)
            ax.set_ylabel('Indice de Demanda',fontsize = 9)
            ax.set_ylim(0,2)
            xticks = ['1','','3','','5','','7','','9','','11','','13','','15','','17','','19','']
            xticks = [1,3,5,7,9,11,13,15,17,19]
            ax.xaxis.set_ticks(xticks)
            axt.xaxis.set_ticks(xticks)
            axt.set_ylim(0,1.3)
            #ax.set_xlim(0.5,13+0.5)
            #ax.grid(axis = 'both' ,color='gray', linestyle='--', linewidth=0.5,which='both')
            if sku==1 and tienda == 1:
                ax.plot(x, D/max,color=extra_color[-2], ls = '--',lw = 2,alpha=1.0,label = 'Forecast')
                ax.plot(x,RD/max,color=f_color[1], lw = 1,alpha=0.9,label = r'Forecast con $\sigma=$'+str(STD))
                ax.plot(x, Xmarker/max,color=extra_color[4], ms = 8,ls = '',marker = "X",alpha=1.0,label = 'Quiebre de stock')
            else:
                ax.plot(x, D/max,color=extra_color[-2], ls = '--',lw = 2,alpha=1.0)
                ax.plot(x,RD/max,color=f_color[1], lw = 1,alpha=0.9)
                ax.plot(x, Xmarker/max,color=extra_color[4], ms = 8,ls = '',marker = "X",alpha=1.0)


            y = (repo+inve)/Lcapacidad[tienda_1]
            axt.bar(x, y, bar_width,lw = 3,edgecolor='white',color=st_color[sku_1][tienda_1],alpha = 0.6,bottom=y_offset)#,label = f'Stock SKU:0{sku} Tienda:0{tienda}')
            axt.set_xlabel('Semana',fontsize = 9)
            axt.set_ylabel('Indice de Ocupación',fontsize = 9)
            axt.set_title(f'Ocupación Tienda: 0{tienda}',fontsize = 10)
            axt.annotate(text[2][tienda_1],xy=(0.93,1.05), xycoords='axes fraction',fontsize=12,color = 'black')
            y_offset = y_offset + y

            #grafico transporte
            axtr = axs[3,1]
            yTr  = repo/capacidad
            axtr.bar(x, yTr, bar_width,lw = 3,edgecolor='white',color=st_color[sku_1][tienda_1],alpha = 0.6,bottom=y_offset_tr)

            axtr.set_ylim(0,1.3)
            axtr.set_xlim(0.5,Nsemanas+0.5)
            axtr.xaxis.set_ticks(xticks)
            axtr.set_xlabel('Semana',fontsize = 9)
            axtr.set_ylabel('Indice de Transporte',fontsize = 9)
            axtr.set_title(f'Reposición semanal',fontsize = 10)
            axtr.annotate(text[3][1],xy=(0.93,1.05), xycoords='axes fraction',fontsize=12,color = 'black')
            y_offset_tr = y_offset_tr + yTr

            #grafico scd

            if tienda == 1:
                axScd = axs[3,0]
                yScd  = scd_model[sku_1,:-1]/SCD0[sku]
                print(x.shape,yScd.shape)
                axScd.bar(x, yScd, bar_width,lw = 3,edgecolor='white',color=s_color[sku_1],alpha = 0.6,bottom=y_offset_scd,label=f'Stock CD SKU 0{sku}')
                axScd.set_ylim(0,2.3)
                axScd.set_xlim(0.5,Nsemanas+0.5)
                axScd.xaxis.set_ticks(xticks)
                axScd.set_xlabel('Semana',fontsize = 9)
                axScd.set_ylabel('Indice de Stock CD',fontsize = 9)
                axScd.set_title(f'Stock en centro de distribución',fontsize = 10)
                axScd.annotate(text[3][0],xy=(0.93,1.05), xycoords='axes fraction',fontsize=12,color = 'black')
                y_offset_scd = y_offset_scd + yScd
    axtr.plot(x, np.ones_like(x),color=extra_color[-3], ls = '--',lw = 2,alpha=1.0,label = 'Transporte máximo')
    #fig.legend(bbox_to_anchor=[0.5, 0.02],loc='center', ncol=4)
    fig.legend(loc='lower center', ncol=4)
    #fig.legend(ncol = 4,loc='center left', bbox_to_anchor=(-0.01, -0.4))
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
                        [11990 ,10990]])

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
Mdemanda    = np.array([[200 ,250 ],
                        [150 ,100]])
#Mme         = Mdemanda
Npeak       = 8
tamano      = 4

#trasporte
capacidad   = 2000
#capacidad   = np.sum(Mme) + 1500

#stock conjunto maximo en tiendas
Lcapacidad  = np.array([1000, 800])#*2

#inventario inicial periodo cero
Minv        = np.array([[0 ,0 ],
                        [0 ,0]])
#Minv        = Mme

#stock en centro de distribucion
SCD0 =  {1:10000,
         2:10000}

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


Fvol = {i:1 for i in SKU}
#parametros auxiliares

'''
*****************************************************
                    Optimizacion
*****************************************************
'''
t1 = time.time()
#parametros temporales, ventana de tiempo
vT  = 8
STD = 0.6
#STD = 0.0
#output_vals = ModeloVariasVentanas(Nsemanas ,vT,SKU ,Ts ,P ,C ,F ,SCD0 ,I0 ,Me ,Tr ,B ,Fvol)       #modelo sin ruido, tamano de ventana modificable
#output_vals = ModeloVariasVentanas1semana(Nsemanas ,SKU ,Ts ,P ,C ,F ,SCD0 ,I0 ,Me ,Tr ,B ,Fvol)   #modelo sin ruido, tamano de ventana 1
output_vals =RuidoGRB(Nsemanas ,vT,SKU ,Ts ,P ,C ,F ,SCD0 ,I0 ,Me ,Tr ,B ,Fvol, STD)                #modelo con ruido ruido gausiano, std seteable


Mvals, scd_model, ocupado_tiendas = obtenerCurvas(output_vals, SKU, Ts, Nsemanas, SCD0, Minv)
#RepartidoT = Mvals[0]
#Transportado = np.sum(np.sum(RepartidoT,axis = 0),axis=0)
t2 = time.time()
print('tiempo de ejecucion: ',round(t2-t1,2))
'''
*****************************************************
                    Ganancias
*****************************************************
'''
#NGsemanales = Nsemanas - vT
NGsemanales = 13
GS, CS = Ganancias(Mvals[2], P, C, SKU, Ts, NGsemanales)#ganancia semanal
G = int(np.sum(GS))#ganancia total en el periodo de Nsemanas
CT= int(np.sum(CS))
#print(GS)
print('Ganancias: ', G)
print('Unidades vendidas: ', CT)
'''
*****************************************************
                    Graficos
*****************************************************
'''
#RF = F
RF = curvaForcastRuido(output_vals['Ruido'],F,SKU,Ts) #forecast con ruido
TodaslasCurvas(Mvals, Minv, F,RF,vT, SKU, Ts, STD,capacidad,scd_model,SCD0,Lcapacidad)
tienda = 1
#plotOcupaTienda(Mvals,Minv,Lcapacidad,tienda)
tienda = 2
#plotOcupaTienda(Mvals,Minv,Lcapacidad,tienda)
#plotSCDhistorico(scd_model,SCD0)
#plotRepoTransporte(Mvals[0], capacidad)
#sku = 2
#plotRepoQuiebres(Mvals, Minv, F, vT, tienda, sku)

#plotResumen4(Mvals, Minv, F,RF,vT, SKU, Ts,STD)
'''
*****************************************************
                    Fin
*****************************************************
'''
qui = Mvals[3,:,:,:12]
rows,cols,weeks = qui.shape
quiebres = rows*cols*weeks - np.sum(qui)

print('N quiebres: ',quiebres)
