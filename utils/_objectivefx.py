# Import modules
import numpy as np
import pandas as pd

def Beneficio1(x ,A ,B ,S ,shape ,factor = 10**(10) ,penalty = 'constant'):
    '''
    Entrega el beneficio de la empresa dada la venta de X productos con precios A

    Parámetros:
    ------------------
    X       : numpy array de 4 indices
            X[N,i,j,t] corresponde a la la partícula N que representa la reposición del SKU i en la tienda j en la semana k
    A       : numpy array de 3 indices
            A[i,j,t] corresponde al eneficio por venta del SKU i en la tienda j en la semana k
    B       : numpy array de 2 indices
            B[j,t] corresponde al stock máximo que puede albergar una tienda j en la semana t, contemplando todos los SKU
    S       : numpy array de 2 indices
            S[i,t] corresponde al máximo stock que se puede reponer en tiendas en la semana t del SKU i
    shape   : tupla de tamaño 3 que da las dimensiones del problema.
    factor  : float, factor por el cual se multiplica la penalizacion
    penalty : str,  Tipo de penalizacion:
              constant es un descuento constante por restriccion violada.
              exponential es proporcional a la exponencial de la posicion de la particula por restriccion violada.
    '''
    #calculo de penalizacion
    if penalty == 'exponential':
        P = np.exp(np.sqrt(np.sum(x**2)))
    else:
        P = 1

    #beneficio
    Nparticles = x.shape[0]
    score = np.zeros(Nparticles)
    for p in range(Nparticles):
        I, J, T = shape
        X = x[p,:].reshape(shape)
        f = np.sum(np.multiply(X,A))
        #restricciones
        penalty = 0
        for t in range(T):
            Xt = X[:,:,t]
            for i in range(I):
                valor_restriccion = np.sum(X[i,:,t])
                if valor_restriccion > S[i,t]:
                    penalty += P * factor
            for j in range(J):
                valor_restriccion = np.sum(X[:,j,t])
                if valor_restriccion > B[j,t]:
                    penalty += P * factor
        score[p] = -f + penalty

    return score

def Beneficio2(x ,A ,B ,S ,shape ,factor = 10**(10) ,penalty = 'constant'):
    '''
    Entrega el beneficio de la empresa dada la venta de X productos con precios A

    Parámetros:
    ------------------
    X       : numpy array de 3 indices
            X[N,j,i*t] corresponde a la la partícula N que representa la reposición del SKU i en la tienda j en la semana k
    A       : numpy array de 2 indices
            A[j,i*t] corresponde al eneficio por venta del SKU i en la tienda j en la semana k
    B       : numpy array de 2 indices
            B[j,t] corresponde al stock máximo que puede albergar una tienda j en la semana t, contemplando todos los SKU
    S       : numpy array de 1 indices
            S[i*t] corresponde al máximo stock que se puede reponer en tiendas en la semana t del SKU i
    shape   : tupla de tamaño 3 que da las dimensiones del problema.
    factor  : float, factor por el cual se multiplica la penalizacion
    penalty : str,  Tipo de penalizacion:
              constant es un descuento constante por restriccion violada.
              exponential es proporcional a la exponencial de la posicion de la particula por restriccion violada.
    '''

    #beneficio
    X = x.reshape(shape)
    revenue = np.sum(np.multiply(X,A))

    #penalizacion por stock a repartir
    stock_a_repartir = np.sum(X,axis=1)

    #penalizacion por stock maximo en tienda
    weeks = B.shape[1]
    skus = int(shape[2]/weeks)
    stock_por_tienda = np.zeros((shape[0],shape[1],weeks))
    for i in range(weeks):
        stock_por_tienda[:,:,i] = np.sum(X[:,:,i*skus:(i+1)*skus],axis=2)

    if penalty == 'linear':
        penalty1 = np.sum((stock_a_repartir > S) * (stock_a_repartir - S) * factor)
        penalty2 = np.sum((stock_por_tienda > B) * (stock_por_tienda - B) * factor)
    elif penalty == 'connstant':
        penalty1 = np.sum((stock_a_repartir > S) * factor)
        penalty2 = np.sum((stock_por_tienda > B) * factor)

    score = penalty1 + penalty2 - revenue
    return score

def beneficio03(X ,R ,S ,params_tiendas ,Npart ,Nsku ,Nt ,Ns):
    #penaltys
    penalty1= np.zeros(Npart)
    penalty2= np.zeros(Npart)
    revenue = np.zeros(Npart)
    factor  = 10**6

    col = ['semana 1','semana 2','semana 3','semana 4','semana 5','semana 6']
    Rdf = pd.DataFrame(R,columns = col)
    Rdf['tienda'] = np.repeat(np.arange(1 ,Nt+1),Nsku)
    Rdf['sku'] = np.resize(np.arange(1,Nsku+1),Nt*Nsku)
    Sdf = pd.DataFrame(S,columns = col)
    Sdf['sku'] = np.arange(1,Nsku+1)

    for p in range(Npart):
        #crear pansas de X (3 columnas de indices, particulas y tiendas y sku) R (tiendas y sku) S(solo sku)

        Xdf = pd.DataFrame(X[p].reshape((Nt*Nsku,Ns)),columns = col)
        Xdf['tienda'] =np.repeat(np.arange(1 ,Nt+1),Nsku)
        Xdf['sku'] = np.resize(np.arange(1,Nsku+1),Nt*Nsku)
        #computo beneficio
        revenue[p] = np.sum( Xdf.values * Rdf.values )

        #computo penalizacion 1
        for i in range(1,Nsku+1):
            cantidad_de_venta = np.cumsum(np.sum(Xdf.loc[Xdf['sku']==i].values ,axis = 0)[:-2])
            stock_acumulado = np.cumsum(Sdf.loc[Sdf['sku']==i].values)[:-1]
            penalty1[p] = penalty1[p] + np.sum((cantidad_de_venta > stock_acumulado) * factor)

        #computo penalizacion 2
        for j in range(1,Nt+1):
            max_repo = params_tiendas[f'IC{j}0']
            repo_en_tienda = (Xdf.loc[Xdf['tienda']==j].values)[:,:-2]
            maxrepo_en_tienda = np.max(repo_en_tienda,axis = 0) / np.sum(repo_en_tienda,axis = 0)
            #penalty2[p] = penalty2[p] + np.sum((maxrepo_en_tienda > max_repo) * factor)

    score = penalty1 + penalty2 - revenue
    return score
