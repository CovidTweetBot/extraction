#Librerias
from datetime import datetime, date

import pandas as pd

from urllib.request import Request, urlopen

#Fecha_local
now = datetime.now()
fecha = (now.strftime('%Y%m%d'))

#Requests
csv_url = 'https://cloud.minsa.gob.pe/s/Md37cjXmjT9qYSa/download'

print('Reading database from link')

req = Request(csv_url, headers={'User-Agent': 'Mozilla/5.0'})

webpage = urlopen(req)

df = pd.read_csv(webpage, sep=';' ,index_col=1, encoding='latin-1')
df_date = df.iloc[-1]['FECHA_CORTE']

print('Fecha del archivo:' + str(df_date))
print('Fecha actual:' + str(fecha))

if fecha == df_date:
  print("Data actualizada, usa la variable df")

else:
  print("Data desactualizada")