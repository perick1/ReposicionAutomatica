# Import modules
import numpy as np

def Beneficio(x,A,B,S,shape):
    '''
    Entrega el beneficio de la empresa dada la venta de X productos con precios A

    Parámetros:
    ------------------
    X : numpy array de 3 indices
        X[i,j,t] corresponde a la reposición del SKU i en la tienda j en la semana k
    A : numpy array de 3 indices
        A[i,j,t] corresponde al eneficio por venta del SKU i en la tienda j en la semana k
    B : numpy array de 2 indices
        B[j,t] corresponde al stock máximo que puede albergar una tienda j en la semana t, contemplando todos los SKU
    S : numpy array de 2 indices
        S[i,t] corresponde al máximo stock que se puede reponer en tiendas en la semana t del SKU i
    '''
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
                    penalty += P
            for j in range(J):
                valor_restriccion = np.sum(X[:,j,t])
                if valor_restriccion > B[j,t]:
                    penalty += P
        score[p] = -f + penalty

    return score
