# Import modules
import numpy as np
import matplotlib.pyplot as plt
import random as rm

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)

np.random.seed(32)
rm.seed(12)

Nsku        = 3
Ntiendas    = 2
Nsemanas    = 6
funcion     = 'beneficio03'

#R12: Revenue (beneficio) por venta en la tienda 1, del sku 2
curvas_BT   = {'R11':('escalon',30,15,3) ,'R12':('rampla',50,-10,3) ,'R13':('valle',15,10,15),
               'R21':('escalon',30,30,3) ,'R22':('rampla',50,  0,4) ,'R23':('valle',15,10,4)}
#S01: stock total a repartir del sku 1
curvas_ST   = {'S01':('escalon',1000,0,1) ,'S02':('rampla',0,1000,1) ,'S03':('valle',3000,3000,3000)}

plot_curvas(curvas_BT ,'dinero')      #curvas de beneficio (revenue)
plot_curvas(curvas_ST ,'stock')      #curvas de stock a repartir

#SC10: stock constrain (restriccion de stock total), distribucion a tiendas, fraccion del total de stock que va a la tienda 1
#IC10: (Inside constrain) restriccion dentro de la tienda 1, da maximo de la fraccion de la tienda que se puede albergar 1 solo sku
restriccion_del_total_de_stock = {'SC10': 0.6 ,'SC20': 0.4}
restriccion_dentro_de_tienda   = {'IC10': 0.4 ,'IC20': 0.6}

Niteraciones = 100
Nparticulas  = 200
PSO_params   = {'c1': 1.0, 'c2': 1.0, 'w':0.5}

#hacer pso por 6 semanas, devuelve el objeto optimizador ya optimizado
PSO = pso6(Nsku=Nsku ,Nt=Ntiendas ,Ns=Nsemanas ,f=funcion ,R=curvas_BT ,S=curvas_ST
          ,SC=restriccion_del_total_de_stock ,IC=restriccion_dentro_de_tienda ,iter=Niteraciones
          ,Nparticles = Nparticulas ,PSO_params=PSO_params)
