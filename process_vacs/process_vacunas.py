import numpy as np
import os
!pip install seaborn --upgrade
import seaborn as sns
import matplotlib.pyplot as plt
import random
random.seed(10)
from datetime import timedelta

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

# Graficos
# Tipo de Grafico 1 - Vacunas aplicadas (o acumuladas) (por dia o por semana) dependiendo de la edad
def fig_aplicadas_por_edad(depa, aux, width = 0.7, titulo = 'Vacunados'):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    #ax.plot(aux.index, aux['Total'])
    ax.bar(aux.index, aux[9], width, color='b')
    ax.bar(aux.index, aux[8], width, bottom= aux[9], color='g')
    ax.bar(aux.index, aux[7], width, bottom= aux[8] + aux[9], color='r')
    ax.bar(aux.index, aux[6], width, bottom= aux[7] + aux[8] + aux[9], color='c')
    ax.bar(aux.index, aux[5], width, bottom= aux[6] + aux[7] + aux[8] + aux[9], color = 'm')
    ax.bar(aux.index, aux[4], width, bottom= aux[5] + aux[6] + aux[7] + aux[8] + aux[9], color = 'y')
    ax.bar(aux.index, aux[3], width, bottom= aux[4] + aux[5] + aux[6] + aux[7] + aux[8] + aux[9], color = 'k')
    ax.bar(aux.index, aux[2], width, bottom= aux[3] +aux[4] + aux[5] + aux[6] + aux[7] + aux[8] + aux[9], color = (0.44, 0.2, 0.68))
    ax.grid(color = (0,0,0), linestyle = '-', axis = 'y')
    ax.set_ylabel(titulo)
    ax.set_title(depa)
    plt.xticks(rotation = 90)
    # 2 -> 18-19, # 3 -> 20-29, # 4 -> 30-39, # 5 -> 40-49
    # 6 -> 50-59, #7 -> 60-69, #8 -> 70-79, #9 -> 80+
    ax.legend(labels=[ '80+', '70-79', '60-69', '50-59', '40-49', '30-39', '20-29', '18-19'])
    plt.show()

# Tipo de grafico 2 - Vacunas aplicadas por tipo de vacuna (diara o semanal)
def fig_aplicadas_por_tipo_vacuna(depa, aux_2, width = 0.7, titulo = 'Dosis'):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(aux_2.index, aux_2[1], width, color = (0.164, 0.341, 0.514))
    ax.bar(aux_2.index, aux_2[2], width, bottom = aux_2[1], color = 'r')
    ax.grid(color = (0,0,0), linestyle = '-', axis = 'y')
    ax.set_ylabel(titulo)
    ax.set_title(depa)
    ax.legend(labels= ['Primera_Dosis', 'Segunda_Dosis'])
    plt.xticks(rotation = 90)
    plt.show()

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

def graficar_datos(lugar, df_c = df_c, width = 0.7, ubi = 'depa', periodo = 'dia'):   
    aux, aux_cum, aux_2, aux_2_cum = dfs_importantes(lugar, df_c, ubi = ubi, periodo = periodo)
    # Graficar
    #fig_distribucion_edad(depa, aux)
    fig_aplicadas_por_edad(lugar, aux)
    fig_aplicadas_por_edad(lugar, aux_cum, titulo = 'Vacunados acumulados')
    fig_aplicadas_por_tipo_vacuna(lugar,aux_2)
    fig_aplicadas_por_tipo_vacuna(lugar, aux_2_cum, titulo = 'Dosis acumuladas')
	
# PERU

def read_vacunas():
    vacunas = pd.read_excel('VACUNAS_LLEGADAS.xlsx')
    fechas = vacunas['Fecha_llegada'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d').date())
    vacunas = vacunas.set_index(fechas)
    vacunas.drop(['Fecha_llegada'], axis = 1, inplace = True)
    return vacunas

def crear_cums_sem(df_c, vacunas):
    df_aux_c = df_c.copy()
    df_aux_c.loc[: , ('SEMANA_DEL_AÑO')] = df_aux_c['FECHA_VACUNACION'].apply(lambda x: semana_del_año(x))
    vacunas['FECHA_VACUNACION'] = vacunas.index
    vacunas['FECHA_VACUNACION'] = vacunas['FECHA_VACUNACION'].apply(lambda x: semana_del_año(x))
    vacunas_aux = pd.pivot_table(vacunas, index = 'FECHA_VACUNACION', aggfunc = 'sum' )
    vacunas_aux = vacunas_aux.cumsum()/2.0
    primero = vacunas_aux.first_valid_index()
    ultimo = vacunas_aux.last_valid_index()
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

    anterior = 0
    while primero != aumentar_semana(ultimo):
        if primero in vacunas_aux.index:
            anterior = vacunas_aux.loc[primero, 'Cantidad']
        else:
            vacunas_aux.loc[primero, 'Cantidad'] = anterior

        primero = aumentar_semana(primero)

    aux_vacunas = vacunas_aux.sort_index()
    aux_1_cum['Quedan'] = aux_vacunas['Cantidad'] - aux_1_cum['Total']
    aux_2_cum['Quedan'] = aux_vacunas['Cantidad'] - aux_2_cum['Total']
    
    return aux_1_cum, aux_2_cum
    
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
    aux_vacunas = vacunas_s.cumsum()/2.0
    primero = aux_vacunas.first_valid_index()
    ultimo = aux_1_cum.last_valid_index()
    while primero <= ultimo:
        if primero in aux_vacunas.index:
            anterior = aux_vacunas.loc[primero, 'Cantidad']
        else:
            aux_vacunas.loc[primero, 'Cantidad'] = anterior

        primero += timedelta(days = 1)

    aux_vacunas = aux_vacunas.sort_index()
    aux_1_cum['Quedan'] = aux_vacunas['Cantidad'] - aux_1_cum['Total']
    aux_2_cum['Quedan'] = aux_vacunas['Cantidad'] - aux_2_cum['Total']
    
    return aux_1_cum, aux_2_cum

def aumentar_semana(text):
    if int(text[-2:]) + 1 < 10:
        return text[:-2] + str(0) + str(int(text[-2:]) + 1)
    return text[:-2] + str(int(text[-2:]) + 1)

def graficar_acumulador(aux, titulo):
    width = 0.7
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    #ax.plot(aux.index, aux['Total'])
    ax.bar(aux.index, aux[9], width, color='b')
    ax.bar(aux.index, aux[8], width, bottom= aux[9], color='g')
    ax.bar(aux.index, aux[7], width, bottom= aux[8] + aux[9], color=(0.63,0.08,0 ))
    ax.bar(aux.index, aux[6], width, bottom= aux[7] + aux[8] + aux[9], color='c')
    ax.bar(aux.index, aux[5], width, bottom= aux[6] + aux[7] + aux[8] + aux[9], color = 'm')
    ax.bar(aux.index, aux[4], width, bottom= aux[5] + aux[6] + aux[7] + aux[8] + aux[9], color = 'y')
    ax.bar(aux.index, aux[3], width, bottom= aux[4] + aux[5] + aux[6] + aux[7] + aux[8] + aux[9], color = (1, 0.3, 0))
    ax.bar(aux.index, aux[2], width, bottom= aux[3] + aux[4] + aux[5] + aux[6] + aux[7] + aux[8] + aux[9], color = (0.44, 0.2, 0.68))
    ax.bar(aux.index, aux['Quedan'], width, bottom = aux[2] + aux[3] + aux[4] + aux[5] + aux[6] + aux[7] + aux[8] + aux[9], color = 'k')
    ax.grid(color = (0,0,0), linestyle = '-', axis = 'y', which = 'both')
    ax.set_ylabel('Vacunados Acumulados (millones)')
    ax.set_title(titulo)
    plt.xticks(rotation = 90)
        # 2 -> 18-19, # 3 -> 20-29, # 4 -> 30-39, # 5 -> 40-49
        # 6 -> 50-59, #7 -> 60-69, #8 -> 70-79, #9 -> 80+
    ax.legend(labels=[ '80+', '70-79', '60-69', '50-59', '40-49', '30-39', '20-29', '18-19', 'Por aplicar'])
    plt.show()
    
def fig_aplicadas_por_tipo_PERU(depa, aux_2,titulo = 'Vacunas acumuladas'):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(aux_2.index, aux_2[1], width, color = (0.164, 0.341, 0.514))
    ax.bar(aux_2.index, aux_2[2], width, bottom = aux_2[1], color = 'r')
    ax.bar(aux_2.index, aux_2['Quedan'], width, bottom = aux_2[1] + aux_2[2], color = 'k')
    ax.grid(color = (0,0,0), linestyle = '-', axis = 'y')
    ax.set_ylabel(titulo)
    ax.set_title(depa)
    ax.legend(labels= ['Primera_Dosis', 'Segunda_Dosis', 'Por aplicar'])
    plt.xticks(rotation = 90)
    plt.show()    

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

def graficar_datos_PERU( df_c = df_c, width = 0.7, periodo = 'dia'):   
    aux, aux_cum, aux_2, aux_2_cum = dfs_normalize_PERU(df_c, periodo = periodo)
   
    fig_aplicadas_por_edad('Vacunados por edad - Peru', aux)
    vacunas = read_vacunas()
    if periodo == 'dia':
        aux_1d_cum, aux_2d_cum = crear_cums_dia(df_c, vacunas)
        graficar_acumulador(aux_1d_cum , 'Primera dosis acumuladas - Peru')
        graficar_acumulador(aux_2d_cum, 'Segunda dosis acumuadas - Peru')
    elif periodo == 'sem':
        aux_1d_cum, aux_2d_cum = crear_cums_sem(df_c, vacunas)
        graficar_acumulador(aux_1d_cum , 'Primera dosis acumuladas - Peru')
        graficar_acumulador(aux_2d_cum, 'Segunda dosis acumuadas - Peru')
    #fig_aplicadas_por_edad('PERU', aux_cum, titulo = 'Vacunados acumulados')
    fig_aplicadas_por_tipo_vacuna('Vacunados por tipo de vacuna - PERU',aux_2)
    
    aux_tot = pd.concat([aux_1d_cum , aux_2d_cum], axis = 1)
    aux_tot = aux_tot.apply(lambda row: row.fillna(aux_1d_cum.loc[row.index]['Total'] + aux_1d_cum.loc[row.index]['Quedan']) )
    aux_2_cum['Quedan'] = np.sum(aux_tot['Quedan'] , axis = 1) 
    
    fig_aplicadas_por_tipo_PERU('Vacunas acumuladas por tipo de vacuna - PERU', aux_2_cum)
	
df_c = vacunas_normalize(df)
graficar_datos('LIMA', periodo = 'sem')
graficar_datos_PERU(periodo = 'sem')