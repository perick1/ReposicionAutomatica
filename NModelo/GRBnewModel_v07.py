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

def ModeloRepoGRB(SKU ,Ts ,T ,P ,C ,D ,SCD0 ,I0 ,Me ,Tr ,B ,F1 ,F2 ,Fvol ,semana):
    #combinaciones
    comb    = [(i,j,t) for i in SKU for j in Ts for t in T]
    combSCD = [(i,t) for i in SKU for t in T]
    #crear modelo
    model = Model('Repo')
    model.Params.LogToConsole = 0
    #agregar variables
    R   = model.addVars(comb ,vtype=GRB.INTEGER ,name='R')
    V   = model.addVars(comb ,vtype=GRB.INTEGER ,name='V')
    I   = model.addVars(comb ,vtype=GRB.INTEGER ,name='I')
    SCD = model.addVars(combSCD ,vtype=GRB.INTEGER ,name='SCD')
    opt = model.addVars(comb ,vtype=GRB.BINARY ,name='opt')
    #funcion objetivo
    print(comb)
    z  = quicksum(V[i,j,t] * (P[i,j][t] - C[i,j][t]) for i,j,t in comb)
    c1 = quicksum(I[i,j,t] * Fvol[i] * F1[t] for i,j,t in comb)
    c2 = quicksum(SCD[i,t] * Fvol[i] * F2[t] for i,t in combSCD)
    model.setObjective(z - c1 - c2 ,GRB.MAXIMIZE)

    #restricciones

    #reposicion no negativa
    model.addConstrs(R[i,j,t] >= 0 for i,j,t in comb)
    #cumplir minimos de exhibicion
    model.addConstrs(I[i,j,t] >= Me[i,j][t] for i,j,t in comb)
    #model.addConstrs((opt[i,j,t] == 1) >> (I[i,j,t] >= Me[i,j][t]) for i in SKU for j in Ts for t in T)
    #reparticion no supera el stock disponible en CdD
    model.addConstrs(quicksum(R[i,j,t] for j in Ts for t in T) <= SCD[i,semana] for i in SKU)
    #no superar maximo almacenaje en tiendas
    model.addConstrs(quicksum(R[i,j,semana] + I0[i,j] for i in SKU) <= B[j][semana] for j in Ts)
    model.addConstrs(quicksum(R[i,j,t] + I[i,j,t-1] for i in SKU) <= B[j][t] for j in Ts for t in T if t!=semana)
    #continuidad de inventario
    model.addConstrs(R[i,j,semana] + I0[i,j] - V[i,j,semana] - I[i,j,semana] == 0 for i in SKU for j in Ts)
    model.addConstrs(R[i,j,t] + I[i,j,t-1] - V[i,j,t] - I[i,j,t] == 0 for i,j,t in comb if t!=semana)
    #continuidad de stock en CD
    #model.addConstrs(SCD[i,semana] + SCD0[i] - quicksum(R[i,j,semana] for j in Ts) == 0 for i in SKU)
    #model.addConstrs(SCD[i,t] + SCD[i,t-1] - quicksum(R[i,j,t] for j in Ts) == 0 for i,t in combSCD if t!=semana)

    #restricciones logicas
    model.addConstrs(V[i,j,semana] <= I0[i,j] + R[i,j,semana] for i in SKU for j in Ts)
    model.addConstrs(V[i,j,t] <= I[i,j,t-1] + R[i,j,t] for i,j,t in comb if t!=semana)
    model.addConstrs(V[i,j,t] <= D[i,j][t] for i,j,t in comb)

    model.addConstrs((opt[i,j,semana] == 0) >> (V[i,j,semana] >= I0[i,j] + R[i,j,semana]) for i in SKU for j in Ts)
    model.addConstrs((opt[i,j,t] == 0) >> (V[i,j,t] >= I[i,j,t-1] + R[i,j,t]) for i,j,t in comb if t!=semana)
    model.addConstrs((opt[i,j,t] == 1) >> (V[i,j,t] >= D[i,j][t]) for i,j,t in comb)

    #Restriccion limite de trasporte semanal
    model.addConstrs(quicksum(R[i,j,t] * Fvol[i] for i in SKU for j in Ts) <= Tr[t] for t in T)
    model.optimize()


    #if T[0]==1:
        #for v in model.getVars():
        #    print(str(v.VarName)+' = '+str(round(v.x,2)))
        #for v in R.values():
            #print("{}: {}".format(v.varName, v.X))
    vals_repo  = { k : v.X for k,v in R.items() }
    vals_inve  = { k : v.X for k,v in I.items() }
    vals_venta = { k : v.X for k,v in V.items() }
    return {'R':vals_repo , 'I' : vals_inve ,'V' : vals_venta}

def plot4(SKU ,Ts ,T ,Tinv,D ,I0, MDL_vals, MDL_vals_old):
    if T[0] ==10:
        fig, axs = plt.subplots(nrows=len(SKU) ,ncols=len(Ts) ,figsize = (10,8))
    dic = {}
    final = {}
    for i in SKU:
        for j in Ts:
            y_offset = np.zeros(5)
            x_offset = 0.25
            bar_width = 0.4

            repo4 = [MDL_vals['R'][i,j,t] for t in T]
            venta = [MDL_vals['V'][i,j,t] for t in T]
            inv4  = [MDL_vals['I'][i,j,t] for t in T]
            inv5  = [I0[i,j]] + inv4
            #d     = [D[i,j][t] for t in T]
            #guardo valores historicos
            dic[(i,j)] = {'R': MDL_vals_old[i,j]['R'] + [repo4[0]],
                          'I': MDL_vals_old[i,j]['I'] + [inv5[0]] ,
                          'V': MDL_vals_old[i,j]['V'] + [venta[0]]}
            final[(i,j)] = {'R': MDL_vals_old[i,j]['R'] + repo4,
                          'I': MDL_vals_old[i,j]['I'] + inv5 ,
                          'V': MDL_vals_old[i,j]['V'] + venta}
            if T[0] ==10:
                ax = axs[i-1,j-1]
                ax.fill_between(Tinv,0,500,facecolor='blue', alpha=0.2)

                ax.plot(np.arange(1,len(D[i,j])+1) + 0.5,D[i,j].values(),ls = '-',color = 'red',label = 'demanda')
                ax.fill_between(np.arange(1,len(D[i,j])+1) + 0.5,0,D[i,j].values(),facecolor='red', alpha=0.4)
                #valores historicos
                T_hist = np.arange(1,len(dic[(i,j)]['R'])+1)
                ax.bar(T_hist - x_offset +0.5, dic[(i,j)]['I'], bar_width, color='gray')
                #y_offset_hist = np.array(inv5)
                ax.bar(T_hist - x_offset+ 0.5, dic[(i,j)]['R'], bar_width, bottom=dic[(i,j)]['I'], color='green')
                ax.bar(T_hist + x_offset+ 0.5, dic[(i,j)]['V'], bar_width, color='orange')

                #nueva optimizacion
                ax.bar(Tinv- x_offset +0.5, inv5, bar_width, bottom=y_offset, color='gray',label = 'inventario')
                y_offset = y_offset + np.array(inv5)
                ax.bar(T- x_offset+ 0.5, repo4, bar_width, bottom=inv5[:-1], color='green',label = 'reposicion')
                ax.bar(T+ x_offset+ 0.5, venta, bar_width, color='orange',label ='venta')
                ax.set_title(f'SKU {i} en tienda {j}')
                ax.set_ylim(0,500)
                ax.set_xlim(1,15)
                ax.xaxis.set_ticks([n+1 for n in range(Nsemanas+1)])
                ax.legend()
                ax.grid(axis = 'both' ,color='gray', linestyle='--', linewidth=0.5)
                plt.show(block=False)
                fig.tight_layout()

    return [dic ,final]

def ModeloVariasVentanas(Tt ,vT,SKU ,Ts ,P ,C ,D ,SCD0 ,I0 ,Me ,Tr ,B ,Fvol):
    #tiempo total en semanas
    TT = [ t+1 for t in range(Tt)]
    #otros costos
    F1   = {t:0*t for t in TT}#en tienda
    F2   = {t:1*t for t in TT}#scd
    for sem in TT[:-(vT)]:
        T = np.arange(sem,sem+vT)
        print(T)
        ModeloRepoGRB(SKU ,Ts ,T ,P ,C ,D ,SCD0 ,I0 ,Me ,Tr ,B ,F1 ,F2 ,Fvol ,sem)

def plotQuiebres():
    pass

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
#Mdemanda    = np.array([[300 ,300 ],
#                        [150 ,100]])
Mdemanda    = np.array([[50 ,50 ],
                        [15 ,10]])
Npeak       = 6
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
D       = Demanda(Mdemanda ,Npeak , tamano ,Nsemanas)
C       = Costos(Mcostos ,Nsemanas)
Me      = MinExhibicion(Mme ,Nsemanas)
Tr      = Transporte(capacidad ,Nsemanas)
B       = StockTiendas(Lcapacidad ,Nsemanas)

#tiendas y skus
SKU = [1 ,2]
Ts  = [1 ,2]
#parametros temporales, ventana de tiempo y optimizacion total]
vT  = 8
Tt  = 20

Fvol = {i:1 for i in SKU}
#parametros auxiliares
MDL_vals_old = {(i,j) : {'R':[],'I':[],'V':[]} for i in SKU for j in Ts}
'''
*****************************************************
                    Optimizacion
*****************************************************
'''

ModeloVariasVentanas(Tt ,vT,SKU ,Ts ,P ,C ,D ,SCD0 ,I0 ,Me ,Tr ,B ,Fvol)
'''
SCD_bodega = np.zeros((2,Nsemanas))
for sem in TT[:-3]:
    T    = [sem ,sem+1 ,sem+2 ,sem+3]
    Tinv = [sem ,sem+1 ,sem+2 ,sem+3 ,sem+4]
    print(T)
    #optimizar 4 semans
    MDL_vals = ModeloRepoGRB(SKU ,Ts ,T ,P ,C ,D ,SCD ,I0 ,Me ,Tr ,B ,F1 ,F2 ,Fvol ,sem)
    #Obtener valores para graficar
    T    = np.array(T)
    Tinv = np.array(Tinv)

    MDL_vals_old ,final = plot4(SKU ,Ts ,T ,Tinv,D ,I0, MDL_vals, MDL_vals_old)
    #actualizar valores de inventario y SCD
    I0 = {(i,j) : MDL_vals['I'][i,j,sem] for i in SKU for j in Ts}
    repartido = {i : sum({j: MDL_vals['R'][i,j,sem] for j in Ts}.values()) for i in SKU}
    SCD = {i : SCD[i] - repartido[i] for i in SKU}
    SCD_bodega[0,sem-1] = SCD[1]
    SCD_bodega[1,sem-1] = SCD[2]
    '''
'''
*****************************************************
                    Graficos
*****************************************************
'''
'''
t2 = time.time()
print('tiempo de ejecucion: ',round(t2-t1,2))
bar_width = 0.8

color = np.array([['cyan' ,'magenta' ],
                  ['orange' ,'black']])
fig, ax = plt.subplots(figsize = (5,4))
y_offset = np.zeros(Nsemanas)
ax.fill_between([0,15],0,np.sum(Mme),facecolor='red', alpha=0.4)
ax.fill_between([0,15],0,capacidad,facecolor='green', alpha=0.3)
for i in SKU:
    for j in Ts:
        y = final[i,j]['R']
        x = np.arange(1,len(y)+1)
        ax.bar(x+0.5, y, bar_width, bottom=y_offset,alpha = 0.5, color=color[i-1,j-1],label = f'SKU {i} en tienda {j}')
        y_offset = y_offset + y

ax.set_title('Reposiciones por semana')
ax.set_ylim(0,capacidad*1.2)
ax.set_xlim(0,14)
ax.xaxis.set_ticks([n+1 for n in range(Nsemanas+1)])
ax.legend()
ax.grid(axis = 'both' ,color='gray', linestyle='--', linewidth=0.5)
plt.show(block=False)
fig.tight_layout()
'''

'''
*****************************************************
                    Fin
*****************************************************
'''
