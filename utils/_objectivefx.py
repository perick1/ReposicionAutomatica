# Import modules
import numpy as np

def Beneficio(x ,A ,B ,S ,shape ,factor = 10**(10) ,penalty = 'constant'):
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
        P = 10**(20)
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
