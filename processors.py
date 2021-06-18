import pathlib
from datetime import datetime

import pandas as pd


def quitar_no_años(row):
    if row['TIEMPO EDAD'] != 'AÑOS' or row['EDAD'] == 'SIN REGISTRO':
        return 0
    else:
        return int(row['EDAD'])


def a_fecha(fecha):
    return datetime.strptime(fecha, '%Y-%m-%d').date()


def quitar_neg(x):
    return max(0, x)


def sinadef_normalize(df):
    df_c = df.copy()
    bins = [-1, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, float("inf")]
    df_c['FECHA_FALLECIMIENTO'] = df_c['FECHA_FALLECIMIENTO'].apply(
        lambda x: datetime.strptime(str(x), '%Y%m%d').date())
    # 0 -> 0 ,1 -> 1 - 10, 2 -> 11 - 20, 3 -> 21- 30, 4 -> 31 - 40
    # 5 -> 41 - 50, 6 -> 51 - 60, 7 -> 61 - 70, 8 -> 71 - 80, 9 -> 81 - 90, 10 -> 91+
    df_c['Edad_fallecimiento'] = pd.cut(df_c['EDAD_DECLARADA'], bins)
    df_c['Edad_fallecimiento'] = df_c['Edad_fallecimiento'].cat.codes
    return df_c


def sinadef_por_dia(df, columna):
    df_counts = pd.pivot_table(df, index=['FECHA_FALLECIMIENTO'], columns=columna, aggfunc='size')
    df_counts = df_counts[:-1]
    df_counts = df_counts.fillna(0)
    return df_counts


def sinadef_por_semana(df, columna):
    df_aux = df.copy()
    df_aux.loc[:, 'SEMANA_DEL_AÑO'] = df_aux['FECHA_FALLECIMIENTO'].apply(lambda x: (x.isocalendar()[:2]))
    df_return = pd.pivot_table(df_aux, index=['SEMANA_DEL_AÑO'], columns=columna, aggfunc='size')
    df_return = df_return.fillna(0)
    return df_return


def apply_processing(df):
    output_path = pathlib.Path("./output")
    results_path = output_path / "results"
    df = sinadef_normalize(df)
    lista = ['SEXO', 'Edad_fallecimiento', 'DEPARTAMENTO']
    for columna in lista:
        data_frame = sinadef_por_dia(df, columna)
        data_frame.to_csv(results_path / f'Por_dia_{columna}.csv')
        data_frame_semana = sinadef_por_semana(df, columna)
        data_frame_semana.to_csv(results_path / f'Por_semana_{columna}.csv')
