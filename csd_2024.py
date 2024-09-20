#!/usr/bin/env python
# coding: utf-8

# # MAPA SINÓPTICO - IDEAM
# ## LIBRERIAS
# ### General, MLS (H/L), Precipitable water, 
from datetime import datetime, timezone
from metpy.plots import ctables
from metpy import plots
from metpy.plots import add_metpy_logo, add_timestamp
from PIL import Image
## get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.offsetbox import AnchoredText
from matplotlib.figure import Figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import ImageTk, Image
from datetime import datetime
from xarray.backends import NetCDF4DataStore
import xarray as xr
from siphon.catalog import TDSCatalog
from siphon.catalog import TDSCatalog
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from metpy.units import units
from netCDF4 import num2date
import numpy as np
from scipy.ndimage import gaussian_filter
# from scipy.ndimage import maximum_filter, minimum_filter
# from siphon.ncss import NCSS
import matplotlib.pyplot as plt
# from scipy.ndimage.filters import maximum_filter, minimun_filter
import pytz
from datetime import datetime, timedelta, timezone
from matplotlib.patheffects import withStroke

# ### Station plot
import urllib.request
import os
import requests
from bs4 import BeautifulSoup
import metpy.io.metar as metar
from metpy.io import metar
# from metar_functions import parse_metar_file
from metpy import plots
from metpy.plots import add_metpy_logo, current_weather, sky_cover, StationPlot
from metpy.calc import reduce_point_density
from metpy.io import metar

# ## Ondas
import pandas as pd
import xlrd
import openpyxl
import numpy as np
from pyproj import Geod

# ## Frentes
from metpy.cbook import get_test_data
from metpy.io import parse_wpc_surface_bulletin
from metpy.plots import (add_metpy_logo, ColdFront, OccludedFront, StationaryFront, StationPlot, WarmFront)
import urllib.request
from metpy.io import parse_wpc_surface_bulletin
from io import BytesIO
# from scipy.ndimage import maximum_filter, minimum_filter
import pandas as pd
from shapely.wkt import loads
from shapely.geometry import Point, LineString  # Importar los tipos de geometría necesarios


 
# # MSLP and 1000-500 hPa Thickness with High and Low Symbols
# 
# 
# Plot MSLP, calculate and plot 1000-500 hPa thickness, and plot H and L markers.
# Beyond just plotting a few variables, in the example we use functionality
# from the scipy module to find local maximum and minimimum values within the
# MSLP field in order to plot symbols at those locations.
# 

# ## Function for finding and plotting max/min points
def plot_maxmin_points(lon, lat, data, extrema, nsize, symbol, color='k',
                       plotValue=True, transform=None, ax=None):
    """
    This function will find and plot relative maximum and minimum for a 2D
    grid. The function can be used to plot an H for maximum values (e.g.,
    High pressure) and an L for minimum values (e.g., low pressue). It is
    best to used filetered data to obtain  a synoptic scale max/min value.
    The symbol text can be set to a string value and optionally the color
    of the symbol and any plotted value can be set with the parameter color

    lon = plotting longitude values (2D)

    lat = plotting latitude values (2D)

    data = 2D data that you wish to plot the max/min symbol placement

    extrema = Either a value of max for Maximum Values or min for Minimum
    Values

    nsize = Size of the grid box to filter the max and min values to plot a
    reasonable number

    symbol = String to be placed at location of max/min value

    color = String matplotlib colorname to plot the symbol (and numerica
    value, if plotted)

    plot_value = Boolean (True/False) of whether to plot the numeric value
    of max/min point

    ax = axes object to plot onto, defaults to current axes

    The max/min symbol will be plotted only within the bounding frame
    (i.e., clip_on=True, clip_box=ax.bbox)
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import maximum_filter, minimum_filter
    # from scipy.ndimage import maximum_filter, minimum_filter


    if ax is None:
        ax = plt.gca()

    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')

    mxy, mxx = np.where(data_ext == data)

    for i in range(len(mxy)):
        ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]], symbol, color=color,
                size=24, clip_on=True, clip_box=ax.bbox,
                horizontalalignment='center', verticalalignment='center',
                transform=transform)
        ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]],
                '\n' + str(int(data[mxy[i], mxx[i]])),
                color=color, size=12, clip_on=True, clip_box=ax.bbox,
                fontweight='bold', horizontalalignment='center',
                verticalalignment='top', transform=transform)


# # Get siphon a TDSCatalog
from siphon.catalog import TDSCatalog
best_gfs = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/'
                      'Global_0p25deg/catalog.xml?dataset=grib/NCEP/GFS/Global_0p25deg/Best')
#best_gfs.datasets

best_ds = list(best_gfs.datasets.values())[0]
ncss = best_ds.subset()
#ncss.variables

from datetime import datetime, timezone
# Crear un objeto de fecha y hora consciente del huso horario en UTC
now_utc = datetime.now(timezone.utc)
now_utc
# Utilizar el objeto de fecha y hora en la consulta
# query.lonlat_box(north=70, south=-25, east=0, west=-150).time(now_utc)

query = ncss.query()
query.all_times()
query.add_lonlat()
# query.lonlat_box(north=70, south=-25, east=0, west=-150).time(datetime.utcnow())
query.lonlat_box(north=70, south=-25, east=0, west=-150).time(now_utc)
query.accept('netcdf4')
query.variables('Precipitable_water_entire_atmosphere_single_layer', "Pressure_reduced_to_MSL_msl", 
                "Geopotential_height_isobaric", "u-component_of_wind_isobaric", "v-component_of_wind_isobaric", 
                "Temperature_isobaric").add_lonlat().accept('netcdf')

# import xarray as xr
#from xarray.backends import NetCDF4DataStore
#from netCDF4 import num2date
data = ncss.get_data(query)
ds = xr.open_dataset(NetCDF4DataStore(data)).metpy.parse_cf()
#list(ds)


# # Extract data into variables
# ## Precipitable water
Precipitable_water = ds['Precipitable_water_entire_atmosphere_single_layer']
pw = Precipitable_water.metpy.unit_array.squeeze()

#datetime.now().strftime('%Y-%m-%d')
ds = ds.squeeze().set_coords(['longitude', 'latitude'])                          
# Grab pressure levels
plev = list(ds.isobaric.values)
# Grab lat/lons and make all lons 0-360
lats = ds.latitude.values
lons = ds.longitude.values
#lons[lons < 0] = 360 + lons[lons < 0]
lons, lats = np.meshgrid(ds['longitude'],ds['latitude'])

#add_timestamp(ax,time=f ,y=0.04, pretext='Created: ', high_contrast=False, time_format='%d %B %Y %H:%MZ')


# ## Geopotential Heigth
emsl_var = ds.Pressure_reduced_to_MSL_msl.metpy.sel()
mslp = gaussian_filter(emsl_var.metpy.convert_units('hPa'), sigma=3.0)

hght_1000 = ds.Geopotential_height_isobaric.metpy.sel(
    vertical=1000 * units.hPa)
hght_500 = ds.Geopotential_height_isobaric.metpy.sel(
    vertical=500 * units.hPa)

thickness_1000_500 = gaussian_filter(hght_500 - hght_1000, sigma=3.0)


# ## Vientos
level = 850 * units.hPa
uwnd_850 = ds['u-component_of_wind_isobaric'].metpy.sel(
    vertical=level).squeeze().metpy.unit_array
vwnd_850 = ds['v-component_of_wind_isobaric'].metpy.sel(
    vertical=level).squeeze().metpy.unit_array


# # Directorio de trabajo

# Directorio de trabajo
# dir = r"O:/Mi unidad/OSPA/04. Administrativo/02. Contratistas/11. ALEXANDER_MARTINEZ/Contrato_713_2023/Sinoptica"
# dir = r"C:\Users\acercompu\OneDrive - Ideam\OSPA_\04. Administrativo\02. Contratistas\11. ALEXANDER_MARTINEZ\Contrato_713_2023\Sinoptica"
dir = r"C:\Users\Siprot\OneDrive - Ideam\OSPA_\04. Administrativo\02. Contratistas\11. ALEXANDER_MARTINEZ\Contrato_713_2023\Sinoptica"

# ## Logo Ideam
# Se debe encontrar en la ruta del script o especificar la ruta con un formtao **.jpg**

insumos = dir + '/in'

image1 = Image.open(insumos + '/Logo_ideam.jpg')
image1 = image1.resize((100,100))


# ## Fechas, tiempos de datos y ejecución
creado = datetime.now(timezone.utc).strftime('%d %B %Y %H:%MZ')
creado

# Supongamos que ds es tu objeto xarray.Dataset
try:
    # Intenta convertir la coordenada "time" a datetime64[ms]
    time_values = ds.reftime.values.astype('datetime64[ms]')
except AttributeError:
    try:
        # Si la conversión falla debido a que no existe "time", intenta con "time1"
        time_values = ds.reftime1.values.astype('datetime64[ns]')
    except AttributeError:
        # Si ambas conversiones fallan, imprime un mensaje de error
        print("No se encontró la coordenada 'time' o 'time1' en el objeto 'ds'.")
        time_values = None

# Continúa con el código solo si se encontró la coordenada de tiempo
if time_values is not None:
    # Convierte los valores de fecha y hora a objetos datetime de Python
    fecha_datos_datetime = pd.to_datetime(time_values)
    # Convierte los objetos datetime a cadena de texto con el formato '%d %B %Y %H:%MZ'
    fecha_datos_string = fecha_datos_datetime.strftime('%d %B %Y %H:%MZ')
    # Añade 6 horas a los valores de fecha y hora
    fecha_datos_validos_datetime = fecha_datos_datetime + pd.Timedelta(hours=12)
    # Convierte los nuevos objetos datetime a cadena de texto con el mismo formato
    fecha_datos_validos_string = fecha_datos_validos_datetime.strftime('%d %B %Y %H:%MZ')

# Continúa con el resto del código usando las variables "fecha_datos_string" y "fecha_datos_validos_string"
# Aquí puedes agregar el resto de tu código que dependa de las fechas y tiempos convertidos.


# # Datos METAR
prueba_hora_actual = datetime.now(timezone.utc)- timedelta(hours=1)
prueba_hora_formateada = prueba_hora_actual.strftime('%Y%m%d_%H00')
today = prueba_hora_formateada 

# Fecha %Y%m%d
# today = datetime.now().date().strftime("%Y%m%d")+'_1600.txt'
today_link = 'metar_'+ f'{today}'
today_link
# URL del catálogo
url = 'https://thredds-dev.unidata.ucar.edu/thredds/catalog/noaaport/text/metar/catalog.html'

# Realizar solicitud HTTP y parsear HTML
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Encontrar el último archivo disponible
files = soup.find_all('a', href=True)
# latest_file = max([f['href'] for f in files if 'metar_' in f['href']]) ##Maxima fecha
# current_metar = latest_file.split('/')   ## para tomar lo ultino de un link

link_busqueda = soup.find_all("a", string=lambda text: f"{today_link}" in text.lower())

# Definir variables de la url 
base_url = "https://thredds-dev.unidata.ucar.edu/thredds/fileServer/noaaport/text/metar/"

archivo_reciente = f"{today_link}" + ".txt"

# Construir la url del metar
url_metar = f"{base_url}{archivo_reciente}"

# Nueva ruta de descarga del archivo
salida = dir + "/out"
nombre_archivo_metar = os.path.join(salida, f"{today_link}.txt")

# Descargar el archivo
# nombre_archivo_metar = f"{today_link}"
urllib.request.urlretrieve(url_metar, nombre_archivo_metar)

# Obtener el último segmento de la URL como nombre de archivo
# os.path.basename(url_metar)

# # Ruta de salida del archivo
# salida = dir + "/out"

# nombre_archivo_metar = salida + f"/{today_link}.txt"

# # Descarga de archivo
# urllib.request.urlretrieve(url_metar, nombre_archivo_metar)

#f"{nombre_archivo_metar}"

with open(f"{nombre_archivo_metar}", 'r') as f:
    data_metar = metar.parse_metar_file(f)

# Set up the map projection
proj = ccrs.LambertConformal(central_longitude=-95, central_latitude=35,
                             standard_parallels=[35])

# Use the Cartopy map projection to transform station locations to the map and
# then refine the number of stations plotted by setting a 300km radius
point_locs = proj.transform_points(ccrs.PlateCarree(), data_metar['longitude'].values,
                                   data_metar['latitude'].values)
data_metar = data_metar[reduce_point_density(point_locs, 300000.)]


# # Ondas
# Especifica la ruta del archivo .xlsx
##ruta_archivo = r"C:\Users\acercompu\OneDrive - Ideam\OSPA_\01. Tematicas\01. Meteorologia\01. Productos\05. Registros_diarios\RegistroDiarioOndasCicloresTropicales2024.xlsx"
##ruta_archivo = r"G:\Mi unidad\OSPA\01. Tematicas\01. Meteorologia\01. Productos\05. Registros_diarios\RegistroDiarioOndasCicloresTropicales2024.xlsx"
ruta_archivo = r"O:\Mi unidad\OSPA\01. Tematicas\01. Meteorologia\01. Productos\05. Registros_diarios\RegistroDiarioOndasCicloresTropicales2024.xlsx"


# Cargar el DataFrame
dataframe = pd.read_excel(ruta_archivo, engine='openpyxl', sheet_name=0)

# Conservar solo las columnas de la 1 a la 10
dataframe = dataframe.iloc[:, 1:11]

# # Eliminar el encabezado y las dos primeras filas
dataframe = dataframe.iloc[2:].reset_index(drop=True)

# Cambiar los nombres de las columnas
nuevo_orden_columnas = [
    'Fecha', 'Número de la onda', 'Latitud norte', 'Longitud norte',
    'Latitud sur', 'Longitud sur', 'Actividad convectiva',
    'Afectaciones Colombia', 'Alertas NHC','Observaciones'
]

dataframe.columns = nuevo_orden_columnas

# Eliminar filas duplicadas
dataframe = dataframe.drop_duplicates()

# Restablecer los índices
dataframe.reset_index(drop=True, inplace=True)

# from datetime import datetime

try:
    # Formatea la fecha actual al formato '%Y/%m/%d'
    fecha_actual = datetime.now().strftime('%Y/%m/%d')

    # Convierte la fecha actual a un objeto Timestamp de Pandas
    fecha_actual = pd.to_datetime(fecha_actual)

    # Convierte toda la columna 'Fecha' a objetos Timestamp de Pandas
    dataframe['Fecha'] = pd.to_datetime(dataframe['Fecha'])

    # Calcula la diferencia entre cada fecha y la fecha actual
    diferencia = (dataframe['Fecha'] - fecha_actual).abs()

    # Encontrar el índice de la fecha más cercana
    indice_fecha_cercana = diferencia.idxmin()

    # Filtrar el DataFrame para obtener solo las filas con la fecha más cercana o igual a la fecha actual
    filas_cercanas = dataframe[dataframe['Fecha'] == dataframe.loc[indice_fecha_cercana, 'Fecha']]

    print(filas_cercanas)

except Exception as e:
    print(f"Se ha producido un error: {e}")


# # Creando la carpeta y la ruta para guardar el mapa

# Obtén el año actual y el número del mes actual:
# import datetime
now = datetime.now()
year = now.year
month = now.strftime('%m')
# Crea el diccionario para mapear los nombres de los meses en español
meses = {
    '01': '01. ENERO',
    '02': '02. FEBRERO',
    '03': '03. MARZO',
    '04': '04. ABRIL',
    '05': '05. MAYO',
    '06': '06. JUNIO',
    '07': '07. JULIO',
    '08': '08. AGOSTO',
    '09': '09. SEPTIEMBRE',
    '10': '10. OCTUBRE',
    '11': '11. NOVIEMBRE',
    '12': '12. DICIEMBRE'
}

# Crea la ruta de la carpeta principal:
##ruta_principal = r"C:\Users\acercompu\OneDrive - Ideam\OSPA_\01. Tematicas\01. Meteorologia\01. Productos\06. Situacion_Sinoptica/"
#ruta_principal = r"G:\Mi unidad\OSPA\01. Tematicas\01. Meteorologia\01. Productos\06. Situacion_Sinoptica/"
ruta_principal = r"O:\Mi unidad\OSPA\01. Tematicas\01. Meteorologia\01. Productos\06. Situacion_Sinoptica/"


ruta_principal = os.path.join(ruta_principal, str(year))

# Verifica si la carpeta principal existe, y si no, créala
if not os.path.exists(ruta_principal):
    os.makedirs(ruta_principal)

# Crea la subcarpeta "Diaria_doc"
ruta_diaria_doc = os.path.join(ruta_principal, "Diaria_doc")
if not os.path.exists(ruta_diaria_doc):
    os.makedirs(ruta_diaria_doc)

# Crea la subcarpeta del mes actual
ruta_mes = os.path.join(ruta_diaria_doc, meses[month])
if not os.path.exists(ruta_mes):
    os.makedirs(ruta_mes)

# Crea la subcarpeta "Carta_sinoptica"
ruta_carta = os.path.join(ruta_mes, "Carta_sinoptica")
if not os.path.exists(ruta_carta):
    os.makedirs(ruta_carta)

# Genera el nombre del archivo utilizando la fecha actual
fecha_actual = now.strftime('%Y%m%d_%H%M')
nombre_archivo_jpg = f"Carta_sinoptica_{fecha_actual}HLC.jpg"

# Combina la ruta de la carpeta "Carta_sinoptica" con el nombre del archivo
ruta_archivo_jpg = os.path.join(ruta_carta, nombre_archivo_jpg)

# Normaliza la ruta para utilizar el separador "/"
ruta_archivo_jpg = os.path.normpath(ruta_archivo_jpg)

nombre_mes_espanol = meses[now.strftime('%m')]
fecha_datos_validos_datetime = fecha_datos_datetime + pd.Timedelta(hours=12)
# Formatea la fecha
fecha_datos_string_espanol = fecha_datos_datetime.strftime('%H:%MZ %d') +' DE' + nombre_mes_espanol[3:]+ fecha_datos_datetime.strftime(' %Y')
# Obtener un día adelante
fecha_manana = fecha_datos_datetime + timedelta(days=1)
# Formatear la fecha resultante
fecha_manana_string = fecha_manana.strftime('%H:%MZ %d') + ' DE' + nombre_mes_espanol[3:] + fecha_manana.strftime(' %Y')
# Obtener la fecha y hora actual en UTC
creado_utc = datetime.now(timezone.utc)
# Formatear la fecha y hora actual en español
creado_espanol = creado_utc.strftime('%H:%MZ %d') + ' DE '+ nombre_mes_espanol[3:]  + creado_utc.strftime(' %Y')

# print(fecha_manana_string)
# print(fecha_datos_string_espanol)

# # Fronts

# [Unified Surface Analysis](https://ams.confex.com/ams/pdfpapers/124199.pdf)

# [Archivo en formato kml](https://ocean.weather.gov/gis/UA_SFC_ANAL.kml)

# url = 'https://www.wpc.ncep.noaa.gov/discussions/codsus'
url = 'https://www.wpc.ncep.noaa.gov/discussions/codsus'

with urllib.request.urlopen(url) as response:
    content = response.read()

df = parse_wpc_surface_bulletin(BytesIO(content))

# Lee el archivo Excel
##excel_file = r'C:\Users\acercompu\OneDrive - Ideam\OSPA_\01. Tematicas\01. Meteorologia\01. Productos\05. Registros_diarios\Sistemas_Frontales.xlsx'
# excel_file = r'G:\Mi unidad\OSPA\01. Tematicas\01. Meteorologia\01. Productos\05. Registros_diarios\Registro_Sistemas_Frontales_2024.xlsx'
excel_file = r'O:\Mi unidad\OSPA\01. Tematicas\01. Meteorologia\01. Productos\05. Registros_diarios\Registro_Sistemas_Frontales_2024.xlsx'


excel_df = pd.read_excel(excel_file)

# Combina los datos del archivo Excel con los datos meteorológicos
combined_df = pd.concat([df, excel_df], ignore_index=True)


# In[42]:


from shapely.geometry import Point, LineString  # Importar los tipos de geometría necesarios

# Verificar si los valores en la columna 'geometry' son objetos geométricos
# Si ya son objetos geométricos, no es necesario convertirlos
def convert_to_geometry(x):
    if isinstance(x, (Point, LineString)):
        return x
    else:
        return loads(x)

# Convertir las cadenas de texto de la columna 'geometry' en objetos geométricos
combined_df['geometry'] = combined_df['geometry'].apply(convert_to_geometry)



# # PLOT MAPA SINOPTICO
# ## Proyection
# get_ipython().run_line_magic('matplotlib', 'inline')
# Set projection of map display
mapproj = ccrs.PlateCarree()
# Set projection of data
dataproj = ccrs.PlateCarree()

# ruta_archivo_jpg

# ## Plot
fig = plt.figure(figsize=(20.,14.))
ax = plt.subplot(111, projection=mapproj)
gl = ax.gridlines(
        draw_labels=True,
        linewidth=1,
        color='gray',
        alpha=0.5,
        linestyle='--'
    )

####################################################################################
#                         STATION PLOT                                             #
####################################################################################
# Change the DPI of the resulting figure. Higher DPI drastically improves the
# look of the text rendering.
plt.rcParams['savefig.dpi'] = 255

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 12 pt.
stationplot = StationPlot(ax, data_metar['longitude'].values, data_metar['latitude'].values,
                          clip_on=True, transform=ccrs.PlateCarree(), fontsize=7)

# Plot the temperature and dew point to the upper and lower left, respectively, of
# the center point. Each one uses a different color.
stationplot.plot_parameter('NW', data_metar['air_temperature'].values, color='red')
stationplot.plot_parameter('SW', data_metar['dew_point_temperature'].values,
                           color='darkgreen')

# A more complex example uses a custom formatter to control how the sea-level pressure
# values are plotted. This uses the standard trailing 3-digits of the pressure value
# in tenths of millibars.
stationplot.plot_parameter('NE', data_metar['air_pressure_at_sea_level'].values,
                           formatter=lambda v: format(10 * v, '.0f')[-3:])

# Plot the cloud cover symbols in the center location. This uses the codes made above and
# uses the `sky_cover` mapper to convert these values to font codes for the
# weather symbol font.
stationplot.plot_symbol('C', data_metar['cloud_coverage'].values, sky_cover)

# Same this time, but plot current weather to the left of center, using the
# `current_weather` mapper to convert symbols to the right glyphs.
stationplot.plot_symbol('W', data_metar['current_wx1_symbol'].values, current_weather)

# Add wind barbs
stationplot.plot_barb(data_metar['eastward_wind'].values, data_metar['northward_wind'].values)

# Also plot the actual text of the station id. Instead of cardinal directions,
# plot further out by specifying a location of 2 increments in x and 0 in y.
stationplot.plot_text((2, 0), data_metar['station_id'].values)

########################################################################################
#                                 ONDAS
#######################################################################################
# import pandas as pd
# import numpy as np
# from pyproj import Geod
# import matplotlib.pyplot as plt

# Crear listas vacías para almacenar las coordenadas
latitud = []
longitud = []

# Crear un objeto Geod para realizar cálculos geodésicos
geod = Geod(ellps='WGS84')

# Iterar sobre las filas del DataFrame
for index, fila in filas_cercanas.iterrows():
    fecha_valor = fila['Fecha']

    # Verificar si la fecha_valor no es nula
    if pd.notnull(fecha_valor):
        try:
            fecha_dt = pd.to_datetime(fecha_valor)
            fecha_formateada = fecha_dt.strftime('%Y/%m/%d')
        except ValueError:
            print(f"La cadena '{fecha_valor}' no es una fecha válida.")
            continue  # Saltar a la siguiente iteración del bucle
    else:
        print("La fecha es nula.")
        continue  # Saltar a la siguiente iteración del bucle

    # Manejar posibles errores al convertir las coordenadas a números
    try:
        lat_norte = float(fila['Latitud norte'][:-1])
        lat_sur = float(fila['Latitud sur'][:-1])
        lon_este = float(fila['Longitud norte'][:-1])
        lon_oeste = float(fila['Longitud sur'][:-1])
    except (ValueError, TypeError):
        print(f"Error al convertir coordenadas en la fila {index}")
        continue  # Saltar a la siguiente iteración del bucle

    # Ajustar las coordenadas según las direcciones cardinales
    if fila['Longitud norte'][-1] == 'W':
        lon_este *= -1
    if fila['Longitud sur'][-1] == 'W':
        lon_oeste *= -1
    if fila['Latitud norte'][-1] == 'S':
        lat_norte *= -1
    if fila['Latitud sur'][-1] == 'S':
        lat_sur *= -1

    # Calcular los puntos intermedios en la línea geodésica
    num_puntos_intermedios = 50
    puntos_intermedios = geod.npts(lon_este, lat_norte, lon_oeste, lat_sur, num_puntos_intermedios)
    lon_intermedios, lat_intermedios = zip(*puntos_intermedios)

    # Calcular el punto medio de la línea
    punto_medio_lat = (lat_norte + lat_sur) / 2
    punto_medio_lon = (lon_este + lon_oeste) / 2

    # Trazar la línea semicurva para la fila actual
    plt.plot([lon_este] + list(lon_intermedios) + [lon_oeste],
             [lat_norte] + list(lat_intermedios) + [lat_sur],
             color='black', linewidth=1.5)

    # Agregar coordenadas y detalles de la onda a las listas
    latitud.append(lat_norte)
    latitud.append(lat_sur)
    longitud.append(lon_este)
    longitud.append(lon_oeste)
    #onda_numeracion = int(fila['Número de la onda'])
    #onda_nombre = f"Onda\ntropical\n{onda_numeracion}"


    # Obtener el valor de la onda
    valor_numero_onda = fila['Número de la onda']

    # Verificar si el valor es un número o una cadena de texto
    if isinstance(valor_numero_onda, int):
    # Si es un número, formatearlo como el nombre de la onda
        onda_nombre = f"Onda\ntropical\n{valor_numero_onda}"
    else:
    # Si es una cadena de texto, usarla directamente como el nombre de la onda
        onda_nombre = valor_numero_onda
    
    
    # Colocar el texto en el punto medio de la línea
    contorno = withStroke(linewidth=3, foreground='black')
    plt.text(punto_medio_lon, punto_medio_lat, onda_nombre, fontsize=8,
             color='white', fontweight='bold', ha='center', va='center',
             path_effects=[contorno])

# Resto del código para configurar el mapa...
# plt.show()

        
    # ax.text(punto_medio_lon, punto_medio_lat, onda_nombre, transform=ccrs.PlateCarree(),
    #         horizontalalignment='center', verticalalignment='center', fontsize=8,
    #         bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3',alpha=0.5),
    #         color='white')
#####################################################################################

# Variable agua precipitable
contours = ax.contourf(lons, lats, pw, 200, transform=ccrs.PlateCarree(),cmap='jet',alpha=0.9)
cbar = fig.colorbar(contours, orientation='vertical', aspect=50, shrink=0.5, pad=0.05, label = "Agua total precipitable (mm)",
                  extendrect='True', format="%.0f",ticks=[0,10,20,30,40,50,60,70])
cbar.ax.set_yticklabels([0,10,20,30,40,50,60,70])

# Plot thickness with multiple colors
clevs = (np.arange(0, 5400, 60),
         np.array([5400]),
         np.arange(5460, 7000, 60))
colors = ('tab:blue', 'b', 'tab:red')
kw_clabels = {'fontsize': 11, 'inline': True, 'inline_spacing': 5,
              'fmt': '%i', 'rightside_up': True,
              'use_clabeltext': True}
for clevthick, color in zip(clevs, colors):
    cs = ax.contour(lons, lats, thickness_1000_500,
                    levels=clevthick, colors=color,
                    linewidths=0.7, linestyles='dashed',
                    transform=dataproj)
    plt.clabel(cs, **kw_clabels)

# Plot MSLP
clevmslp = np.arange(800., 1120., 4)
cs2 = ax.contour(lons, lats, mslp, clevmslp, colors='k', linewidths=0.7,
                 linestyles='solid', transform=dataproj)
#Se eliminó *kw_clabels
plt.clabel(cs2)

# Use definition to plot H/L symbols
plot_maxmin_points(lons, lats, mslp, 'max', 50, symbol='H', color='b',
                   transform=dataproj)
plot_maxmin_points(lons, lats, mslp, 'min', 25, symbol='L', color='red',
                   transform=dataproj)

#Logo metpy - Ideam
# plots.add_metpy_logo(fig, x=1120, y=30,size='small')


img = Image.open(insumos + '/Logo_ideam.jpg')
image1 = image1.resize((50,50))
imagebox = OffsetImage(image1)
imagebox.image.axes = ax
ab = AnnotationBbox(imagebox, [-7.7, -17.4], pad=0, frameon=False)
ax.add_artist(ab)


#Fechas de datos
#ax.text(-38, -15, f'CARTA SINÓPTICA', fontsize=12)
textbox = dict(boxstyle='round', facecolor='black', alpha=1)
ax.text(-99.5, -19.5, f'CARTA SINÓPTICA\nModelo GFS:{fecha_datos_string_espanol}\nValido hasta: {fecha_manana_string}\nCreado: {creado_espanol}', 
        style='italic',color='white',fontsize=10,fontweight="bold",bbox=textbox)

# Set extent and plot map lines

ax.set_extent([-100, -5, 50, -20], ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
               edgecolor='black', linewidth=2)
ax.add_feature(cfeature.STATES.with_scale('50m'),
               edgecolor='black', linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=2, edgecolor='black')


##############################################################
#                      winds                                 #
##############################################################

# # Plot wind barbs every fifth element
datacrs = ccrs.PlateCarree()
wind_slice = (slice(None, None, 7), slice(None, None, 7))
ax.barbs(lons[wind_slice][0], lats[wind_slice][:,1],
         uwnd_850[wind_slice[0], wind_slice[1]].to('kt').m,
         vwnd_850[wind_slice[0], wind_slice[1]].to('kt').m,
          length=5, pivot='middle', color='green', transform=datacrs)

# Front systems
# ax.plot([-80,-83,-85,-87,-90], [29, 26, 23, 21, 17],
#        path_effects=[ColdFront()],transform=ccrs.PlateCarree())

features = combined_df[combined_df['feature']== 'LOW']
for f in features['geometry']:
    if (-100 <= f.x <= -5) and (-20 <= f.y <= 50):
        ax.text(f.x, f.y, 'L', transform=ccrs.PlateCarree(), color='red', fontsize=16)

features = combined_df[combined_df['feature']== 'HIGH']
for f in features['geometry']:
    if (-100 <= f.x <= -5) and (-20 <= f.y <= 50):
        ax.text(f.x, f.y, 'H', transform=ccrs.PlateCarree(), color='blue', fontsize=16)
    
s=7
feature_names = ['WARM','COLD','STNRY','OCFNT','TROF']
feature_styles = [{'linewidth':1, 'path_effects':[WarmFront(size=s)]},
                  {'linewidth':1, 'path_effects':[ColdFront(size=s)]},
                  {'linewidth':1, 'path_effects':[StationaryFront(size=s)]},
                  {'linewidth':1, 'path_effects':[OccludedFront(size=s)]}, 
                  {'linewidth':2, 'linestyle':'dashed', 'edgecolor':'darkorange'}]
                                  
for name, style in zip(feature_names, feature_styles):
    f = combined_df[combined_df['feature'] == name]
    ax.add_geometries(f.geometry, crs=ccrs.PlateCarree(), **style, facecolor='none')


# Put on some titles
plt.title('MSLP (hPa) with Highs and Lows, 1000-500 hPa Thickness (m),Precipitable water (mm) and Wind Barbs (850 hPa)', loc='left')
# Ajusta automáticamente el diseño para evitar superposiciones
fig.tight_layout()
#plt.title(f'VALID: {vtime}', loc='right')
# plt.show()
######################################################################################
# Guarda la figura en la ruta especificada
plt.savefig(ruta_archivo_jpg, dpi=300, bbox_inches='tight')


# # Sistemas frontales - Tabla guia
# import pandas as pd
# from datetime import datetime

# Obtener la fecha actual
current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Crear el DataFrame con los datos proporcionados
tabla_frentes = {
    'valid': [current_date] * 8,
    'feature': ['HIGH', 'LOW', 'COLD', 'WARM', 'OCFNT', 'STNRY', 'TROF', 'COLD'],
    'strength': ['1024', '985', 'WK', 'WK', 'WK', 'WK', '', 'WK'],
    'geometry': [
        'POINT (-77 64)',
        'POINT (-75 51)',
        'LINESTRING (-68 43, -69 37, -71 35, -73 31, -76 29, -76 29, -78 27)',
        'LINESTRING (-67 49, -64 50, -61 50)',
        'LINESTRING (-75 51, -75 51, -72 51, -68 49, -67 49)',
        'LINESTRING (-79 69, -87 71, -94 72, -108 74, -122 76, -132 77)',
        'LINESTRING (-77 54, -81 57, -86 60)',
        'LINESTRING (-63 45, -63 35, -66 30, -70 25, -80 20)']
}

df = pd.DataFrame(tabla_frentes)

# Guardar el DataFrame en un archivo Excel
##file_path = r'C:\Users\acercompu\OneDrive - Ideam\OSPA_\01. Tematicas\01. Meteorologia\01. Productos\05. Registros_diarios\Sistemas_Frontales.xlsx'
#file_path = r'G:\Mi unidad\OSPA\01. Tematicas\01. Meteorologia\01. Productos\05. Registros_diarios\Registro_Sistemas_Frontales_2024.xlsx'
file_path = r'O:\Mi unidad\OSPA\01. Tematicas\01. Meteorologia\01. Productos\05. Registros_diarios\Registro_Sistemas_Frontales_2024.xlsx'


df.to_excel(file_path, index=False)

print("Archivo Excel guardado con éxito en:", file_path)


# # Cerrar datos NetCDF
data.close()







