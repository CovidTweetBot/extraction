#Created by Sebastian Gomez
import numpy as np
import os
#!pip install seaborn --upgrade
import seaborn as sns
import matplotlib.pyplot as plt
import random
random.seed(10)
from datetime import timedelta
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import plotly.graph_objects as go
import plotly.express as px

import collections
from random import choices
def semana_del_año(x):
    aux = x.isocalendar()[:2]
    if len(str(aux[1])) >= 2:
        return str(aux[0]) + '_' + str(aux[1])
    if len(str(aux[1])) < 2:
        return str(aux[0]) + '_0' + str(aux[1])
    
def a_fecha(txt):
    return datetime.datetime.strptime(txt, '%Y-%m-%d').date()

#Librerias
from datetime import datetime, date

import pandas as pd

from urllib.request import Request, urlopen

#Fechas
today = date.today()
today = today.strftime("%Y%m%d")

#Requests
csv_url = "https://cloud.minsa.gob.pe/s/ZgXoXqK2KLjRLxD/download"

print('Reading database from link')

req = Request(csv_url, headers={'User-Agent': 'Mozilla/5.0'})

webpage = urlopen(req)

df = pd.read_csv(webpage)

df_date = df.iloc[-1]['FECHA_CORTE']


if df_date == today:
  print('Datos actualizados y cargados en memoria, utiliza la var df')
else:
  print('Datos aún no han sido actualizados')
  print("Ultima actualización:" + str(df_date))

import datetime
# Normalizando la data
def corregir(texto, typo,  texto_corregido):
    if texto == typo:
        return texto_corregido
    else:
        return texto

def vacunas_normalize(df):
    
    def replace_random(value):
        if value == 0.0:
            return choices(population, weights)[0]
        else:
            return value
    
    df_c = df.copy()
    
    edades_distribucion = df_c['EDAD'].value_counts() / (len(df_c)- np.sum(df_c['EDAD'].isnull()))
    population = np.array(edades_distribucion.index)
    weights = edades_distribucion.values
    
    df_c.fillna(0, inplace = True)
    
    df_c['EDAD'] = df_c['EDAD'].apply(lambda x: replace_random(x)) 
    
    bins = [-1, 0, 10, 20, 30, 40,50,60, 70 , 80 , float("inf")]
    # 2 -> 18-19, # 3 -> 20-29, # 4 -> 30-39, # 5 -> 40-49
    # 6 -> 50-59, #7 -> 60-69, #8 -> 70-79, #9 -> 80+
    df_c['Edad_bin']  = pd.cut(df_c['EDAD'], bins, right = False)
    df_c['Edad_bin'] = df_c['Edad_bin'].cat.codes
    df_c['PROVINCIA'] = df_c['PROVINCIA'].apply(lambda x: corregir(x, 'SAN ROMAS', 'SAN ROMAN'))
    
    # Usamos la informacion de este excel para obtener el codigo ubigeo.
    pobla_c = pd.read_excel('Poblacion por distrito_.xlsx')
    pobla = pobla_c.copy()
    #print([i for i in df_c['DEPARTAMENTO'].unique() if i not in pobla['DEPARTAMENTO'].unique()])
    #print([i for i in df_c['PROVINCIA'].unique() if i not in pobla['PROVINCIA'].unique()])
    #print([i for i in df_c['DISTRITO'].unique() if i not in pobla['DISTRITO'].unique()])
    pobla['UBIGEO_text'] = pobla['DEPARTAMENTO'] + '_' + pobla['PROVINCIA']

    dict_ubigeos = {}
    for i in range(len(pobla)):
        dict_ubigeos[pobla.iloc[i]['UBIGEO_text']] = pobla.iloc[i]['UBIGEO'][:4]

    df_c['UBIGEO_text'] = df_c['DEPARTAMENTO'] + '_' + df_c['PROVINCIA']
    #df_c['UBIGEO_text'] = df_c['DEPARTAMENTO'] + '_' + df_c['PROVINCIA'] + '_' + df_c['DISTRITO']
    df_c['UBIGEO'] = df_c['UBIGEO_text'].apply(lambda x: dict_ubigeos.get(x, 'No_identificado'))
    pobla['Ub_dep'] = pobla['UBIGEO'].apply(lambda x : x[:2])
    df_c['Ub_dep'] = df_c['UBIGEO'].apply(lambda x : (str(x)[:2]))

    df_c['FECHA_VACUNACION'] = df_c['FECHA_VACUNACION'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d').date())

    return df_c

# graficas
def graficas_depa_prov(depa, aux, aux2 = None, titulo = 'Vacunados', grafico = 1):
    fig = go.Figure()
    if grafico == 1:
        columns = [9,8,7,6,5,4,3,2]
        names = [ '80+', '70-79', '60-69', '50-59', '40-49', '30-39', '20-29', '18-19']
        colors = [ 'rgb(0,0,255)', 'rgb(20,180,0)', 'rgb(255,0,0)', 'rgb(0,236,237)',
                 'rgb(247,0,247)', 'rgb(255,242,0)', 'rgb(255,77,0)', 'rgb(123,0,172)']
        assert (len(columns) == len(names) == len(colors))
        for i in range(len(columns)):
            fig.add_trace(
                go.Bar(
                    x = aux.index,
                    y = aux[columns[i]],
                    name = names[i],
                    marker = dict(
                        color = colors[i],
                        line = dict(color = 'rgba(248,248,249)', width = 0.1)
                                 ),
                    )
            )
    
    elif grafico == 2:
        columns = [2,1,]
        names = ['Segunda dosis', 'Primera dosis']
        colors = ['rgb(0,0,255)', 'rgb(255,0,0)']
        assert (len(columns) == len(names) == len(colors))
        for i in range(len(columns)):
            fig.add_trace(
            go.Bar(
                    x = aux.index,
                    y = aux[columns[i]],
                    name = names[i],
                    marker = dict(
                        color = colors[i],
                        line = dict(color = 'rgba(248,248,249)', width = 0.1)
                                 ),
                    )
            )
        
    fig.update_layout(barmode = 'stack')
    fig.update_layout(title_text = titulo + ' ' + depa)
    fig.update_xaxes(title_text = 'Fecha')
    fig.update_yaxes(title_text = 'Vacunas')
    fig.show()
    
def graficar_datos_depa_prov(lugar, df_c, ubi = 'depa', periodo = 'dia'):
    aux, aux_cum, aux_2, aux_2_cum = dfs_importantes(lugar, df_c, ubi = ubi, periodo = periodo)
    graficas_depa_prov(lugar, aux, titulo= 'Vacunas por edades')
    graficas_depa_prov(lugar, aux_cum, titulo = 'Vacunas aplicadas acumuladas por edades')
    graficas_depa_prov(lugar, aux_2, titulo = 'Vacunas por dosis', grafico = 2)
    graficas_depa_prov(lugar, aux_2_cum, titulo = 'Vacunas aplicadas acumuladas por dosis', grafico = 2)
    
def dfs_importantes(ubicacion, df_c, ubi = 'depa', periodo = 'dia'):
    
    if ubi == 'depa':
        df_aux_c = df_c[df_c['DEPARTAMENTO'] == ubicacion]
    elif ubi == 'prov':
        df_aux_c = df_c[df_c['UBIGEO_text'] == ubicacion]
    
    df_aux = df_aux_c.copy()
    
    if periodo == 'dia':
        aux = pd.pivot_table(df_aux , index = 'FECHA_VACUNACION', columns = 'Edad_bin', aggfunc = 'size')
    elif periodo == 'sem':
        df_aux.loc[: , ('SEMANA_DEL_AÑO')] = df_aux['FECHA_VACUNACION'].apply(lambda x: semana_del_año(x))
        aux = pd.pivot_table(df_aux, index = ['SEMANA_DEL_AÑO'], columns = 'Edad_bin', aggfunc = 'size')
    
    aux.fillna(0, inplace = True)
    aux['Total'] = np.sum(aux, axis = 1)
    
    if periodo == 'dia':
        aux_2 = pd.pivot_table(df_aux , index = 'FECHA_VACUNACION', columns = 'DOSIS', aggfunc = 'size')
    elif periodo == 'sem':
        aux_2 = pd.pivot_table(df_aux , index = 'SEMANA_DEL_AÑO', columns = 'DOSIS', aggfunc = 'size')
    aux_2.fillna(0.0, inplace = True)
    aux_cum = aux.drop(['Total'], axis = 1)
    aux_cum = aux.cumsum()
    aux_2_cum = aux_2.cumsum()

    return aux, aux_cum , aux_2, aux_2_cum

# Funcion que Graficar todo
def graficar_plotly_datos_PERU( df_c, periodo = 'dia'):
    aux, aux_cum, aux_2, aux_2_cum = dfs_normalize_PERU(df_c, periodo = periodo)
    graficas('PERU', aux, titulo = 'Vacunados 1ra o 2da')
    
    vacunas = read_vacunas()
    if periodo == 'dia':
        aux_1d_cum, aux_2d_cum = crear_cums_dia(df_c, vacunas)
        graficas('PERU', aux_1d_cum, titulo = 'Primera dosis acumulados - Peru', grafico = 2)
        graficas('PERU', aux_2d_cum, titulo = 'Segunda dosis acumulados - Peru', grafico = 3)
        aux_2d_cum = add_zeros_df(aux_1d_cum, aux_2d_cum)
    elif periodo == 'sem':
        aux_1d_cum, aux_2d_cum = crear_cums_sem(df_c, vacunas)
        graficas('PERU', aux_1d_cum, titulo = 'Primera dosis acumulados - Peru', grafico = 2)
        graficas('PERU', aux_2d_cum, titulo = 'Segunda dosis acumulados - Peru', grafico = 3)
        aux_2d_cum = add_zeros_df(aux_1d_cum, aux_2d_cum, dia = False)
    
    graficas('PERU', aux_2, titulo = 'Vacunas por dosis', grafico = 4)
    graficas('PERU', aux_1d_cum, aux_2d_cum, grafico = 5)
    
# dfs_normalize_PERU
def dfs_normalize_PERU(df_c, periodo = 'dia'):
    
    df_aux = df_c.copy()
    
    if periodo == 'dia':
        aux = pd.pivot_table(df_aux , index = 'FECHA_VACUNACION', columns = 'Edad_bin', aggfunc = 'size')
    elif periodo == 'sem':
        df_aux.loc[: , ('SEMANA_DEL_AÑO')] = df_aux['FECHA_VACUNACION'].apply(lambda x: semana_del_año(x))
        aux = pd.pivot_table(df_aux, index = ['SEMANA_DEL_AÑO'], columns = 'Edad_bin', aggfunc = 'size')
    
    aux.fillna(0, inplace = True)
    aux['Total'] = np.sum(aux, axis = 1)
    
    if periodo == 'dia':
        aux_2 = pd.pivot_table(df_aux , index = 'FECHA_VACUNACION', columns = 'DOSIS', aggfunc = 'size')
    elif periodo == 'sem':
        aux_2 = pd.pivot_table(df_aux , index = 'SEMANA_DEL_AÑO', columns = 'DOSIS', aggfunc = 'size')
    aux_2.fillna(0.0, inplace = True)
    aux_cum = aux.drop(['Total'], axis = 1)
    aux_cum = aux.cumsum()
    aux_2_cum = aux_2.cumsum()

    return aux, aux_cum , aux_2, aux_2_cum

# graficas
# graficas
def graficas(depa, aux, aux2 = None, titulo = 'Vacunados', grafico = 1):
    fig = go.Figure()
    if grafico in [1,2,3,4]:
        if grafico == 1:
            columns = [9,8,7,6,5,4,3,2]
            names = [ '80+', '70-79', '60-69', '50-59', '40-49', '30-39', '20-29', '18-19']
            colors = [ 'rgb(0,0,255)', 'rgb(20,180,0)', 'rgb(255,0,0)', 'rgb(0,236,237)',
                     'rgb(247,0,247)', 'rgb(255,242,0)', 'rgb(255,77,0)', 'rgb(123,0,172)']

        elif grafico == 2:
            columns = [9,8,7,
                       6,5,4,
                       3,2, 'Guardadas',
                      'Quedan']
            names = ['80+', '70-79', '60-69', 
                     '50-59', '40-49', '30-39', 
                     '20-29', '18-19', 'Guardadas 2da dosis',
                    'Por aplicar']
            colors = ['rgb(0,0,255)', 'rgb(20,180,0)', 'rgb(255,0,0)', 
                      'rgb(0,236,237)', 'rgb(247,0,247)', 'rgb(255,242,0)', 
                      'rgb(255,77,0)', 'rgb(123,0,172)', 'rgb(192, 227,0)',
                     'rgb(0,0,0)']

        elif grafico == 3:
            columns = [9,8,7,
                       6,5,4,
                       3,2, 'Por aplicar 2da']
            names = ['80+', '70-79', '60-69', 
                     '50-59', '40-49', '30-39', 
                     '20-29', '18-19', 'Atrasados']
            colors = ['rgb(0,0,255)', 'rgb(20,180,0)', 'rgb(255,0,0)', 
                      'rgb(0,236,237)', 'rgb(247,0,247)', 'rgb(255,242,0)', 
                      'rgb(255,77,0)', 'rgb(123,0,172)', 'rgb(0,0,0)']

        elif grafico == 4:
            columns = [1,2]
            names = ['Primera Dosis', 'Segunda Dosis']
            colors = ['rgb(255,0,0)', 'rgb(0,0,255)']
        
        assert (len(columns) == len(names) == len(colors))
        for i in range(len(columns)):
            fig.add_trace(
                go.Bar(
                    x = aux.index,
                    y = aux[columns[i]],
                    name = names[i],
                    marker = dict(
                        color = colors[i],
                        line = dict(color = 'rgba(248,248,249)', width = 0.1)
                                ),
                    )
            )

    elif grafico == 5:
        fig.add_trace(go.Bar(        
            x = aux.index, y = aux['Total'], name = 'Primera Dosis',
            marker = dict(color = 'rgb(0,0,255)', line = dict(color = 'rgba(248,248,249)', width = 0.1))))
        
        fig.add_trace(go.Bar(
            x = aux.index, y = aux2['Total'], name = 'Segunda Dosis',
            marker = dict(color = 'rgb(255, 0,0)', line = dict(color = 'rgba(248,248,249)', width = 0.1))))
        
        columns = ['Guardadas', 'Quedan']
        names = ['Guardadas 2da dosis', 'Por aplicar']
        colors = ['rgb(192, 227,0)', 'rgb(0,0,0)']    
        for i in range(len(columns)):
            fig.add_trace(
            go.Bar(
                    x = aux.index,
                    y = aux[columns[i]],
                    name = names[i],
                    marker = dict(
                        color = colors[i],
                        line = dict(color = 'rgba(248,248,249)', width = 0.1)
                                 ),
                    )
            )
         
    fig.update_layout(barmode = 'stack')
    fig.update_layout(title_text = titulo + ' ' + depa)
    fig.update_xaxes(title_text = 'Fecha')
    fig.update_yaxes(title_text = 'Vacunas')
    fig.show()
    
# read_vacunas
def read_vacunas():
    vacunas = pd.read_excel('VACUNAS_LLEGADAS.xlsx')
    fechas = vacunas['Fecha_llegada'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d').date())
    vacunas = vacunas.set_index(fechas)
    vacunas.drop(['Fecha_llegada'], axis = 1, inplace = True)
    return vacunas

# crear_cums_dia
def crear_cums_dia(df_c, vacunas):
    df_aux = df_c.copy()
    df_aux_1 = df_aux[df_aux['DOSIS'] == 1]
    df_aux_2 = df_aux[df_aux['DOSIS'] == 2]
    aux_1 = pd.pivot_table(df_aux_1 , index = 'FECHA_VACUNACION', columns = 'Edad_bin', aggfunc = 'size')
    aux_2 = pd.pivot_table(df_aux_2 , index = 'FECHA_VACUNACION', columns = 'Edad_bin', aggfunc = 'size')
    aux_1.fillna(0, inplace = True)
    aux_2.fillna(0, inplace = True)
    aux_1_cum = aux_1.cumsum()
    aux_2_cum = aux_2.cumsum()
    aux_1_cum['Total'] = np.sum(aux_1_cum, axis = 1)
    aux_2_cum['Total'] = np.sum(aux_2_cum, axis = 1)
    aux_1_cum.index[0]
    por_usar = []
    por_usar.append(0)
    vacunas_s = vacunas.groupby(level = 0).sum()
    anterior = 0
    aux_vacunas = vacunas_s.cumsum()
    primero = aux_vacunas.first_valid_index()
    ultimo = aux_1_cum.last_valid_index()
    while primero <= ultimo:
        if primero in aux_vacunas.index:
            anterior = aux_vacunas.loc[primero, 'Cantidad']
        else:
            aux_vacunas.loc[primero, 'Cantidad'] = anterior

        primero += timedelta(days = 1)

    aux_vacunas = aux_vacunas.sort_index()
    aux_2_cum = add_zeros_df(aux_1_cum, aux_2_cum)

    aux_1_cum['Guardadas'] = aux_1_cum['Total'] - aux_2_cum['Total']
    #aux_1_cum['Quedan'] = aux_vacunas['Cantidad'] - aux_1_cum['Total'] - aux_2_cum['Total'] - aux_1_cum['Guardadas']
    aux_1_cum['Quedan'] = aux_vacunas['Cantidad'] - aux_1_cum['Total'] - aux_2_cum['Total'] - aux_1_cum['Guardadas']
    lista = aux_1_cum['Total'].index
    new_lista = lista + timedelta(days = 21)
    add_dataframe = aux_1_cum[['Total', 'Quedan']].set_index(new_lista)
    add_dataframe.drop(['Quedan'], axis = 1, inplace = True)
    add_dataframe.rename(columns = {"Total": "Recibio 1ra"}, inplace = True) # Acumulado de recibió primera hace pasaron 3 semanas
    new_dataframe = pd.concat([aux_2_cum, add_dataframe] , axis = 1)
    new_dataframe['Por aplicar 2da'] = new_dataframe['Recibio 1ra'] - new_dataframe['Total']
    aux_2_cum = new_dataframe[:-21]
    aux_2_cum_new = aux_2_cum.dropna().copy()
    return aux_1_cum, aux_2_cum_new

# crear_cums_sem
def crear_cums_sem(df_c, vacunas):
    df_aux_c = df_c.copy()
    df_aux_c.loc[: , ('SEMANA_DEL_AÑO')] = df_aux_c['FECHA_VACUNACION'].apply(lambda x: semana_del_año(x))
    vacunas['FECHA_VACUNACION'] = vacunas.index
    vacunas['FECHA_VACUNACION'] = vacunas['FECHA_VACUNACION'].apply(lambda x: semana_del_año(x))
    vacunas_aux = pd.pivot_table(vacunas, index = 'FECHA_VACUNACION', aggfunc = 'sum' )
    vacunas_aux = vacunas_aux.cumsum()
    df_aux_1 = df_aux_c[df_aux_c['DOSIS'] == 1]
    df_aux_2 = df_aux_c[df_aux_c['DOSIS'] == 2]
    aux_1 = pd.pivot_table(df_aux_1 , index = 'SEMANA_DEL_AÑO', columns = 'Edad_bin', aggfunc = 'size')
    aux_2 = pd.pivot_table(df_aux_2 , index = 'SEMANA_DEL_AÑO', columns = 'Edad_bin', aggfunc = 'size')
    aux_1.fillna(0, inplace = True)
    aux_2.fillna(0, inplace = True)
    aux_1_cum = aux_1.cumsum()
    aux_2_cum = aux_2.cumsum()
    aux_1_cum['Total'] = np.sum(aux_1_cum, axis = 1)
    aux_2_cum['Total'] = np.sum(aux_2_cum, axis = 1)
    primero = vacunas_aux.first_valid_index()
    ultimo = aux_1_cum.last_valid_index()

    anterior = 0
    while primero != aumentar_semana(ultimo):
        if primero in vacunas_aux.index:
            anterior = vacunas_aux.loc[primero, 'Cantidad']
        else:
            vacunas_aux.loc[primero, 'Cantidad'] = anterior

        primero = aumentar_semana(primero)

    aux_vacunas = vacunas_aux.sort_index()
    aux_2_cum = add_zeros_df(aux_1_cum, aux_2_cum, dia = False)

    aux_1_cum['Guardadas'] = aux_1_cum['Total'] - aux_2_cum['Total']
    aux_1_cum['Quedan'] = aux_vacunas['Cantidad'] - aux_1_cum['Total'] - aux_2_cum['Total'] - aux_1_cum['Guardadas']

    lista = aux_1_cum['Total'].index
    new_lista = [aumentar_semana(aumentar_semana(aumentar_semana(i))) for i in lista]
    add_dataframe = aux_1_cum[['Total', 'Quedan']].copy()
    add_dataframe['New_semana'] = new_lista
    add_dataframe.set_index('New_semana', inplace = True)
    add_dataframe.drop(['Quedan'], axis = 1, inplace = True)
    add_dataframe.rename(columns = {"Total": "Recibio 1ra"}, inplace = True)
    new_dataframe = pd.concat([aux_2_cum, add_dataframe] , axis = 1)
    new_dataframe['Por aplicar 2da'] = new_dataframe['Recibio 1ra'] - new_dataframe['Total']
    aux_2_cum = new_dataframe[:-3]
    aux_2_cum_new = aux_2_cum.dropna().copy()

    return aux_1_cum, aux_2_cum_new
# add_zeros_df
def add_zeros_df(aux_1_cum, aux_2_cum, dia = True):
    # Adding zeros to do the sustraction
    if dia:
        faltantes = list(set(list(aux_1_cum.index)) - set(list(aux_2_cum.index)))
        columns = np.array(aux_2_cum).shape[1]
        zeros = [0]*columns
        for i in faltantes:
            aux_2_cum.loc[i] = zeros
        aux_2_cum.sort_index(inplace = True)
        return aux_2_cum
    elif dia == False:
            # Adding zeros to do the sustraction
        faltantes = list(set(list(aux_1_cum.index)) - set(list(aux_2_cum.index)))
        columns = np.array(aux_2_cum).shape[1]
        zeros = [0]*columns
        for i in faltantes:
            aux_2_cum.loc[i] = zeros
        aux_2_cum.sort_index(inplace = True)
        return aux_2_cum
    
# aumentar_semana
def aumentar_semana(text):
    if int(text[-2:]) + 1 < 10:
        return text[:-2] + str(0) + str(int(text[-2:]) + 1)
    return text[:-2] + str(int(text[-2:]) + 1)

# df_c = vacunas_normalize(df)

# Tipo de grafico 1:
# Departamento
# Por dia
#graficar_datos_depa_prov('PIURA', df_c, ubi = 'depa', periodo = 'dia')

# Tipo de grafico 2:
# Departamento
# Por semana
# graficar_datos_depa_prov('PIURA', df_c, ubi = 'depa', periodo = 'sem')

# Tipo de grafico 3
# Departamento-Provincia
# Por dia
# graficar_datos_depa_prov('PIURA_PIURA', df_c, ubi = 'prov', periodo = 'dia')

# Tipo de grafico 4
# Departamento-Provincia
# Por sem
# graficar_datos_depa_prov('PIURA_PIURA', df_c, ubi = 'prov', periodo = 'sem')

# Tipo de grafico 5 para Peru
# Por dia
# graficar_plotly_datos_PERU(df_c, periodo = 'dia')

# Tipo de grafico 2 para Peru
# Por semana
# graficar_plotly_datos_PERU(df_c, periodo = 'sem')