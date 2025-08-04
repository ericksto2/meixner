
import numpy as np
import pandas as pd # librería para manejo de dataframes
import matplotlib.pyplot as plt
import yfinance as yf # librería para descarga de históricos de Yahoo Finance
import math
import copy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


ticker = 'GFINBURO.MX'
fecha_inicio = '2018-01-01'
fecha_fin = '2025-05-14'
K=100


def calcula_media(ticker, fecha_inicio, fecha_fin):
    df_hist = yf.download(ticker,
                          start = fecha_inicio,
                          end = fecha_fin,
                          progress=False)
    # Cálculo de rendimientos
    rendims = (df_hist.iloc[1:]['Close'] - df_hist.shift(1).iloc[1:]['Close'])/df_hist.shift(1).iloc[1:]['Close']
    # mu
    mu = np.mean(rendims) #media aritmética
    return(mu)

def calcula_desv_est(ticker, fecha_inicio, fecha_fin):
    df_hist = yf.download(ticker,
                          start = fecha_inicio,
                          end = fecha_fin,
                          progress=False)
    # Cálculo de rendimientos
    rendims = (df_hist.iloc[1:]['Close'] - df_hist.shift(1).iloc[1:]['Close'])/df_hist.shift(1).iloc[1:]['Close']
    # sigma
    sigma = np.std(rendims) #desviación estándar muestral
    return(sigma)

def ultimo_precio(ticker, fecha_inicio, fecha_fin): #Para decidir cuál será el precio inicial de la simulación
    df_hist = yf.download(ticker,
                          start = fecha_inicio,
                          end = fecha_fin,
                          progress=False)
    S0 = df_hist.iloc[df_hist.shape[0]-1]['Close']
    return(S0)


def payoff_call(x):
    return(max(x-K,0))

def payoff_put(x):
    return(max(K-x,0))

def vp_un(x):
    return(x*math.exp((-1)*r))

"""### Simulaciones Longstaff y Schwarz"""


## Para generar simulaciones un activo
def genera_simulaciones_individual(ticker1, fecha_inicio, fecha_fin, num_sims, dt, T):
    #Obtención de parámetros
    media1 = float(calcula_media(ticker1, fecha_inicio, fecha_fin))
    desv1 = float(calcula_desv_est(ticker1, fecha_inicio, fecha_fin))

    N = T / dt
    t = np.arange(1, int(N) + 1)

    # Precio inicial de la simulación
    S_01 = float(ultimo_precio(ticker1, fecha_inicio, fecha_fin))

    # Pasos aleatorios primer activo
    b = {str(sim): np.random.normal(0, 1, int(N)) for sim in range(1, num_sims + 1)}

    W = {str(sim): b[str(sim)].cumsum() for sim in range(1, num_sims + 1)}
    drift = (media1 - 0.5 * desv1**2) * t
    proc_varianza = {str(sim): desv1 * W[str(sim)] for sim in range(1, num_sims + 1)}
    S1 = []
    for sim in range(1, num_sims + 1):
      path = S_01 * np.exp(drift + proc_varianza[str(sim)])
      S1.append(np.insert(path, 0, S_01))  # inserta S_01 al inicio de cada trayectoria

    S1 = np.array(S1)

    return(S1)

def genera_df_simulaciones(ticker1, fecha_inicio, fecha_fin, num_sims, dt, T):
    trayectorias = genera_simulaciones_individual(ticker1, fecha_inicio, fecha_fin, num_sims, dt, T)
    df_pr = pd.DataFrame(trayectorias)
    df_pr = df_pr.drop([0], axis = 1, inplace = False)

    nombres_columnas = list(range(1,len(trayectorias[0])))
    nombres_columnas = [str(x) for x in nombres_columnas]
    df_pr.columns = nombres_columnas

    df = df_pr
    return(df)

num_sims = 100
dt = 1
T = 252/2

genera_df_simulaciones(ticker, fecha_inicio, fecha_fin, num_sims, dt, T)



def ejercicio_americana_LS(ticker1, fecha_inicio, fecha_fin, num_sims, dt, T, funcion_payoff, K, r):
    # Generación de trayectorias
    df = genera_df_simulaciones(ticker1, fecha_inicio, fecha_fin, num_sims, dt, T)
    num_cols = df.shape[1]
    num_rengls = df.shape[0]

    # Primera ejecucion (en el penúltimo nodo)
    payoff = df.iloc[:,(num_cols -1)].apply(funcion_payoff)
    indices_ejercicio = list(payoff[payoff>0].index)
    posicion = str(num_cols -1)
    payoff_lag = df[posicion].apply(funcion_payoff)
    indices_ejercer_lag = list(payoff_lag[payoff_lag>0].index)
    x = df.iloc[indices_ejercer_lag,:][posicion]
    r = r
    y = payoff[indices_ejercer_lag].apply(vp_un)
    df_ml = pd.concat([y,x],axis=1)
    df_ml.columns = ['y', 'x']
    X = df_ml['x'].values.reshape(-1, 1)
    y = df_ml['y'].values.reshape(-1, 1)
    polin = PolynomialFeatures(degree = 2, include_bias=True)
    X_polin = polin.fit_transform(X)
    polin.fit(X_polin, y)
    reg_pol = LinearRegression()
    reg_pol.fit(X_polin, y)
    y_predicted =reg_pol.predict(X_polin)
    y_pred_lista = [ item for elem in y_predicted for item in elem]
    y_pred_serie = pd.Series(y_pred_lista)
    y_pred_serie.index = payoff_lag[payoff_lag>0].index
    df_comparacion = pd.concat([payoff_lag[payoff_lag>0],y_pred_serie],axis=1)
    df_comparacion.columns = ['intrinseco', 'continuacion']
    df_comparacion['ejercer'] = np.where(df_comparacion['intrinseco'] >= df_comparacion['continuacion'],True,False)
    lista_ejercer = list(df_comparacion[df_comparacion['ejercer'] == True].index)

    lst = []
    lst.append(indices_ejercicio)
    lst.append(lista_ejercer)

    # Segunda ejecucion
    for m in range(2,num_cols):
        payoff = df.iloc[:,(num_cols - m)].apply(funcion_payoff)
        lista_payoff = list(payoff.index)
        lista_anulados = list(set(lista_payoff) - set(lista_ejercer)) + list(set(lista_ejercer) - set(lista_payoff))
        payoff[lista_anulados] = 0
        posicion = str(num_cols - m)
        payoff_lag = df[posicion].apply(funcion_payoff)
        indices_ejercer_lag = list(payoff_lag[payoff_lag>0].index)
        x = df.iloc[indices_ejercer_lag,:][posicion]
        y = payoff[indices_ejercer_lag].apply(vp_un)
        df_ml = pd.concat([y,x],axis=1)
        df_ml.columns = ['y', 'x']
        X = df_ml['x'].values.reshape(-1, 1)
        y = df_ml['y'].values.reshape(-1, 1)
        polin = PolynomialFeatures(degree = 2, include_bias=True) 
        X_polin = polin.fit_transform(X)
        polin.fit(X_polin, y)
        reg_pol = LinearRegression()
        reg_pol.fit(X_polin, y)
        y_predicted =reg_pol.predict(X_polin)
        y_pred_lista = [item for elem in y_predicted for item in elem]
        y_pred_serie = pd.Series(y_pred_lista)
        y_pred_serie.index = payoff_lag[payoff_lag>0].index
        df_comparacion = pd.concat([payoff_lag[payoff_lag>0],y_pred_serie],axis=1)
        df_comparacion.columns = ['intrinseco', 'continuacion']
        df_comparacion['ejercer'] = np.where(df_comparacion['intrinseco'] >= df_comparacion['continuacion'],True,False)
        lista_ejercer = list(df_comparacion[df_comparacion['ejercer'] == True].index)
        lst.append(lista_ejercer)
    return([lst,df])
  
num_sims = 10
dt = 1
T = 30
funcion_payoff = payoff_put
K_2 = 57
r=0.0923


df_2= ejercicio_americana_LS(ticker, fecha_inicio, fecha_fin, num_sims, dt, T, funcion_payoff, K_2, r)
print(df_2)

