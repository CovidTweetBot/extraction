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