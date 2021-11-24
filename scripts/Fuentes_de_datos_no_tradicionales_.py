# -*- coding: utf-8 -*-
#!pip install geopandas
#!pip install mapclassify
#!pip install contextily
#!pip install geopy
#!pip install libpysal
#!pip install esda
#!pip install imagehash

#!git clone https://github.com/dfbeltran/cursotomadecisiones.git

"""# Fuentes de datos no tradicionales

## 1. Introducción

Es usual que la primera aproximación al análisis de datos se de por la necesidad de entender el comportamiento de un grupo de individuos que se caracterizan a partir de un conjunto de variables. Por ejemplo, la Gran Encuesta Integrada de Hogares (GEIH), desarrollada por el Departamento Administrativo Nacional de Estadística (DANE), captura información a nivel nacional sobre:

* Viviendas: ubicación, tipo y características físicas como materiales de paredes y pisos.
* Hogares: disponibilidad de servicios públicos, privados y comunales, valor pagado y calidad de estos, lugar y energía utilizada para preparar alimentos, tipo de tenencia de la vivienda y tenencia de bienes en el hogar.
* Personas: sexo, edad, parentesco, estado civil, seguridad social en salud, educación, caracteristicas laborales y migración.

Esta información, y mucha de la que normalmente se utliza en el análisis de datos, suele representar al individuo por filas y sus caracter´ısticas por columnas. Adem´as, en muchos casos es posible encontrar entre las variables que definen al individuo datos sobre el lugar con el cual se relaciona, en forma de país, región, ciudad, barrio, parcela, latitud y longitud, etc., añadiendo así un componente espacial que vale la pena aprovechar. Aquí, es importante mencionar que la ubicación no altera la estructura de los datos, donde cada fila sigue representando un individuo y cada columna una variable.

Por su parte, puede suceder que las filas de un conjunto de información representen una relación, como por ejemplo una transacción, un trayecto o un vínculo social, mientras que las columnas indican las características de esta relación, incluyendo las partes que intervienen en la misma. De esta manera, la información tiene una estructura de red que puede aprovecharse para generar conocimiento sobre los indivudos y sus relaciones.

Tomando en consideración lo anterior, esta sección del curso presentaría algunas de las tareas del análisis de información espacial y de vínculos, junto con las metodolog´ıas que apoyan estos tipos de estudios. También, y en primera instancia, se haría una discusión sobre el análisis descriptivo y predictivo.

## 2. Análisis descriptivo, predictivo y prescriptivo

En general, existen tres tipos de análisis: descriptivo, predictivo y prescriptivo. El análisis descriptivo
busca presentar en un lenguaje entendible el resumen de diferentes situaciones que pueden estar presentes
en los datos, pero que no son fáciles de ver sin que exista un procesamiento de por medio. Así, una medida
de tendencia central (promedio, mediana, moda) o de dispersión (varianza, coeficiente de variación) puede
informar sobre el comportamiento de los individuos en torno a una característica determinada. En el caso
multivariado, una correlación indica la asociación entre dos o más variables (o entre dos o más individuos),
y un coeficiente de regresión indica cuanto cambia una variable de interés por cambios en otra variable que
se cree la afecta, manteniendo otras características constantes

Por su parte, el análisis predictivo se preocupa por generar información sobre situaciones que no han sido
observadas aún, como cuando se quiere saber a qué grupo debe asignarse un nuevo individuo, o cuando se
desea conocer el valor de una variable de interés en algún momento futuro.

Finalmente, el análisis prescriptivo pretende dar recomendaciones sobre diferentes alternativas de acción,
informando sobre el resultado mas probable para cada decisión. Para esto se necesita información sobre las
alternativas disponibles y un sistema de retroalimentación que registre el resultado generado por la acción
que se haya tomado.

En esta sección trabajaremos sobre el análisis descriptivo, mientras que posteriormente en el curso nos
ocuparemos de la predicción (y tangencialmente del pronóstico).

## 3. Análisis espacial

### 3.1. Información espacial y su representación gráfica 

Normalmente las caracteristicas de ubicación de un individuo no son suficientes para su análisis geográfico, sino que se necesita, además, de una infraestrucutra que permita referenciar al  individuo con su entorno y en relación con los demás casos observados en el conjunto de datos que se esté trabajando. Es así que se necesita contar con mapas (digitales), los cuales pueden ser de diferentes tipos:

*Líneas: cada una de las unidades ocupa un espacio en 2 dimensiones, que está representado por su
contorno.
* Puntos: cada una de las unidades ocupa un punto en el espacio.
* Polígonos: visualmente muy similar al mapa de líneas, pero en este caso las unidades están representadas por el área que ocupan en 2 dimensiones.

Antes de entrar en detalle, procederemos a descargar e instalar algunos de los paquetes que utiliza Python para el análisis espacial:

> !pip install geopandas

El paquete [geopandas](https://geopandas.org/en/stable/) consiste en un conjunto de herramientas para manipular y leer datos geográficos

Entre sus posibilidades está la de leer mapas digitales que se encuentran en uno de los formatos más populares para este tipo de información: [Shapefiles](https://en.wikipedia.org/wiki/Shapefile) de ESRI

En primera instancia, trabajaremos con el mapa departamental de Colombia. El  archivo encontramos la carpeta llamada Colombia (`/../cursotomadecisiones/datos/tema_3/Colombia`), la cual contiene 6 archivos, todos ellos con el nombre COLOMBIA pero con diferentes extensiones, algunos de ellos necesarios y otros opcionales para contar con un mapa digital completo:
* .shp (obligatorio): almacena las entidades geométricas de los objetos.
* .shx (obligatorio): almacena el índice de las entidades geométricas.
* .dbf (obligatorio): base de datos que almacena la información de los atributos (características) de los objetos.
* .prj (opcional): contiene la información del sistema de coordenadas.
* .sbn y .sbx (opcional): almacena el índice espacial de las unidades.

La carpeta Colombia debe ser almacenada en un una ruta conocida, ya que esta se necesitará para hacer la carga de la información. La lectura del mapa se realiza mediante el comando `read_file`, que utiliza como parámetro el nombre del mapa (en nuestro caso Colombia):
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

mapa = gpd.read_file("/content/cursotomadecisiones/datos/tema_3/Colombia/COLOMBIA.shp")

"""En este caso, se carga el mapa y se almacena en un elemento llamado mapa, el cual es de clase `geopandas.geodataframe.GeoDataFrame`, como se puede ver del comando:"""

type(mapa)

"""El elemento mapa se compone de varios elementos, los cuales se pueden ser llamados al escribir 'mapa['elemento']' En este caso se tienen: polygons, SHAPE_Leng, SHAPE_Area, entre otros. También, existen una serie de funciones que facilitan trabajar con este tipo información como los límites máximos y mínimos que definen un cuadrado donde se encuentra el polígono, área, distancias y otras funciones como determinar si un objeto (punto u otro polígono) se encuentra dentro de otro polígono.

Esto puede verse de la siguiente manera en código:
"""

mapa.head(4)

mapa.bounds.head(4)

# minx, miny, maxx, maxy 
mapa['geometry'].total_bounds

sum(mapa.to_crs(epsg=3395).area)/ 10**6

"""El objeto mapa puede graficarse mediante el comando `plot`, generando el mapa departamental de Colombia en el visor de gráficos:"""

mapa.plot();

"""El método `plot` utiliza la librería `matplotlib` para renderizar las imágenes, es por esto que uno puede personalizar el resultado usando los comandos de esta librería.

La manipulación del objeto `mapa` se puede hacer como cualquier otro DataFrame, para acceder, filtrar, visualizar la información sobre cada uno de los individuos que se componen dentro de este objeto.

Las columnas contienen información sobre el código único de identificación de cada departamento, su año de creación, nombre oficial, acto de ley que lo crea y área, entre otros. 

Se puede generar el mapa asignando un color diferente a cada polígono (departamento en nuestro caso), a partir de la variable `DPTO_CCDGO` y a su vez filtrar la isla de San Andrés y visualizarla por aparte.
"""

san_andres = mapa[mapa['DPTO_CNMBR']=='ARCHIPIELAGO DE SAN ANDRES']
colombia = mapa[mapa['DPTO_CNMBR']!='ARCHIPIELAGO DE SAN ANDRES']

fig, axs = plt.subplots(1,2,figsize=(20,10))

# Crear mapa colombia sin san andres
axs[0].set_title('Colombia')
colombia.plot(ax = axs[0], column = 'DPTO_CCDGO')
axs[0].set_xlabel('Longitude [°]')
axs[0].set_ylabel('Latitude [°]')

# Crear mapa San Andres y Providencia
axs[1].set_title('San Andres y Providencia')
san_andres.plot(ax = axs[1], column = 'DPTO_CCDGO')
axs[1].set_xlabel('Longitude [°]')
axs[1].set_ylabel('Latitude [°]')

plt.suptitle("Mapa Político de Colombia", fontsize=20)
fig.tight_layout(rect=[0, 0.03, 0.9, 0.95])
plt.show();

"""En general, no es fácil entender la dinámica geográfica de un fenómeno al observar los datos que lo representan, y es aquí donde los mapas son de gran utilidad. Por ejemplo, el archivo Delito.csv contiene información sobre homicidios en Colombia para el año 2016, según la información suministrada por el Instituto Nacional de Medicina Legal y Ciencias Forenses. Los datos se ven así:

"""

import pandas as pd

delito = pd.read_csv('/content/cursotomadecisiones/datos/tema_3/Delito.csv', encoding = 'latin-1', sep = ';')
delito.head()

"""Esta información puede ayudar a enteder cuál es el departamento más violento del país, a partir de la medida de homicidios. En primer lugar, al objeto `mapa` se le añadirá la información de homicidios mediante el comando [merge](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html), utilizando la variable identificadora comó un OBJECTID. Luego se genera el mapa de homicidios en Colombia para el año 2016:"""

mapa = mapa.merge(delito, on='OBJECTID')

mapa['escala_color'] = (mapa['Homicidios']-(mapa['Homicidios'].min())) / ((mapa['Homicidios'].max())-(mapa['Homicidios'].min()))

fig, axs = plt.subplots(1,1,figsize=(10,10))
mapa.plot(column = 'escala_color', cmap = 'Blues', edgecolor = 'k', legend = True, ax = axs);

"""Como se puede observar, Valle del Cauca, Antioquia y Bogotá presentan la mayor cantidad de homicidios.

Sin embargo, se debe considerar que estos son los departamnetos con la mayor cantidad de población en el país por lo cual, para responder mejor a la pregunta, se calcularán tasas de homicidio y se representarán de la misma forma que en los casos anteriores
"""

mapa['tasa_100'] = mapa['Homicidios']/mapa['Poblacion']*100000
mapa['escala_color'] = mapa['tasa_100']/(max(mapa['tasa_100'])-min(mapa['tasa_100']))

fig, axs = plt.subplots(1,1,figsize=(10,10))
mapa.plot(column = 'escala_color', cmap = 'Blues', edgecolor = 'k', legend = True, ax = axs);

"""De esta forma cambia el resultado, mostrando valores altos para los departamentos de Valle del Cauca, Quindío, Norte de Santander, Chocó, Arauca y Meta."""

mapa[['DPTO_CNMBR_x','Homicidios','tasa_100']].sort_values('tasa_100', ascending = False).head()

"""
GeoPandas cuenta con los parametros `scheme` y `k` para hacer automaticamente clasificar los poligonos en intervalos, el `scheme` puede ser en k-percentiles, k-intervalos del mismo tamaño y k-divisiones naturales (minimizan la varianza). 

Para nuestro ejemplo podemos clasificar por tercios la tasa de homicidios para asignar 3 Niveles: ALTO, MEDIO, BAJO.
"""

fig, axs = plt.subplots(1,1,figsize=(10,10))
mapa.plot(column = 'escala_color', scheme='equal_interval', k=3, cmap = 'Blues', edgecolor = 'k', legend = True, ax = axs);

"""A continuación se tomarán los datos del archivo Vivienda.csv, que corresponden a ubicaciones y precios de vivienda usada en la ciudad de Medellín. Debido a que para cada individuo se cuenta con su longitud y
latitud, se puede generar a partir de estos datos un mapa de puntos:
"""

vivienda = pd.read_csv('/content/cursotomadecisiones/datos/tema_3/Vivienda.csv', sep = ';', decimal=',')
vivienda.head()

plt.figure(figsize=(7,7))
plt.scatter(x=vivienda['longitud'], y=vivienda['latitud'])
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.show()

"""La representación espacial de esta información se realizará mediante la libreria `contextily` que tiene una integración nativa con `geopandas`, este paquete permite
añadir un mapa base que facilita la referenciación de la información. Para esto se utiliza el comando `add_basemap`. Adicionalmente, se especifican las opciones zoom = 13, el cual controla el acercamiento.

Por lo tanto, primero procederemos a crear un GeoDataFrame a partir de nuestro DataFrame especificando las columnas longitud y latitud para que se creen como una figura geométrica, en este caso puntos.
"""

import contextily as cx
vivienda = gpd.GeoDataFrame(vivienda, geometry=gpd.points_from_xy(vivienda['longitud'], vivienda['latitud']))
vivienda.head()

"""Una vez hemos creado nuestro GeoDataFrame debemos asignarle un sistema de coordenadas, dado que existen varias representaciones para proyectar puntos en un mapa.

Generalmente, los mapas Web están en un sistema de coordenadas 'Web Mercator' (EPSG 3857), pero nuestros datos están en ESPG 4326 (latitud y longitud), por lo que le asignamos a nuestro set de datos esa representación y transformamos al nuevo sistema de coordenadas para finalmente gráficar con `contextily` de la siguiente manera:
"""

vivienda = vivienda.set_crs('epsg:4326', inplace= True)
vivienda = vivienda.to_crs(epsg=3857)

ax = vivienda.plot(figsize=(10, 10), color = 'r', markersize=20)
cx.add_basemap(ax)

"""Por último podemos asignar a cada punto un tamaño y color de acuerdo con su precio"""

fig, ax = plt.subplots(figsize=(15,15))
scatter = vivienda.plot(column = 'precio', cmap='Reds', 
                   markersize=vivienda['precio']/1e6,
                   scheme="quantiles",
                   alpha=0.6, 
                   legend = True,
                   ax = ax,
                   legend_kwds={'fontsize':20,
                                'facecolor':'lightgrey',})
cx.add_basemap(scatter)

leg = ax.get_legend()

leg.set_title('Precio MM', prop = {'size':20})

leg._loc = 3
for lbl in leg.get_texts():
    label_text = lbl.get_text()
    lower = label_text.split()[0].replace(',','')
    upper = label_text.split()[1]
    new_text = f'{float(lower)/1e6:,.0f}-{float(upper)/1e6:,.0f}'
    lbl.set_text(new_text)

leg.set_bbox_to_anchor((1, 0.5, 0.1, 0))

"""A continuación, se considera la construcción de variables que capturan otras dimensiones del comportamiento espacial de cada individuo. Por ejemplo, al determinar una coordenada para el centro de la ciudad es posible calcular la distancia (euclidiana) de cada individuo mediante la fórmula 

<center>$distancia_{ij} = \sqrt{(longitud_{i}-longitud_{j})^2-(latitud{i}-latitud{j})^2}$</center>

donde $i$ y $j$ hacen referencia a dos puntos, representados en el espacio por su longitud y latitud. De esta forma, $i$ puede ser cada uno de los inmuebles mientas que $j$ es el centro de la ciudad. Para obtener la coordanda de una ciudad o dirección se puede usar `geopy`:
"""

# import module
from geopy.geocoders import Nominatim
# initialize Nominatim API
geolocator = Nominatim(user_agent="geoapiExercises")
ubicacion = geolocator.geocode("Medellin")
print(f'latitud: {ubicacion.latitude}, longitud: {ubicacion.longitude}')

"""Ahora, es posible calcular la distancia de cada individuo al centro mediante:"""

vivienda['dist_centro'] = np.sqrt(np.square((vivienda['longitud']- ubicacion.latitude))
                                     + np.square((vivienda['longitud']- ubicacion.longitude)))

"""Esta nueva variable puede utilizarse en una estrategia de modelamiento más avanzada, como las que buscan generar grupos naturales en los datos, identificar observaciones atípicas o predecir alguna caraterística que bajo circusntancias determinadas es no observada para un grupo de individuos.

Para finalizar, cuando se usa información espacial generalmente se trabaja con múltiples capas y aprovechando una herramienta que permite interacción como un notebook, permite profundizar en la información y detalle.  

`Plotly` es una de las herramientas más populares en Python que permite ser desplegada en notebooks e incluso páginas web, a continuación se da un ejemplo en donde se pretende capturar todos los puntos de atención de las oficinas de EPM en Antioquia.

Primero, cargamos los polígonos de Antioquia de la siguiente manera:
"""

antioquia = gpd.read_file("/content/cursotomadecisiones/datos/tema_3/Medellín/MGN_ADM_MPIO_GRAFICO.shp")

fig, ax = plt.subplots(figsize = (10,10))
base = antioquia.plot(color='white', edgecolor='grey', ax = ax)

"""Una vez hemos cargado los polígonos, cargamos las Oficinas de atención al cliente de EPM en Antioquia [datos recuperados de datos abiertos](https://www.datos.gov.co/Funci-n-p-blica/Oficinas-de-atenci-n-al-cliente-EPM-Antioquia-/e3dj-zguu) y podemos poner esa capa de puntos sobre el polígono que visualizamos previamente"""

oficinas_epm = pd.read_csv('/content/cursotomadecisiones/datos/tema_3/Oficinas_de_atenci_n_al_cliente_-_EPM._Antioquia..csv')
oficinas_epm = gpd.GeoDataFrame(oficinas_epm, geometry=gpd.points_from_xy(oficinas_epm['Longitud'], oficinas_epm['Latitud']))

fig, ax = plt.subplots(figsize = (10,10))
base = antioquia.plot(color='white', edgecolor='grey', ax = ax)
oficinas_epm.plot(ax=base, marker='o', color='red', markersize=4);

"""Como la información parece consistente, procedemos a graficar usando `plotly`."""

import plotly.express as px
import json

fig = px.scatter_mapbox(oficinas_epm, lat="Latitud", lon="Longitud", hover_name="Dirección Oficina", hover_data=["Región", "Localidad"],
                        color_discrete_sequence=["fuchsia"], zoom=8, height=500, center = {'lat':ubicacion.latitude, 
                                                                                            'lon':ubicacion.longitude},
                        ).update_traces(marker={"size": 15})

fig.update_layout(mapbox={"style": "open-street-map",
                          "zoom": 9,
                          "center": {'lat':6.16372275,  'lon':-75.71936312},
                          "layers": [{
                                "source": json.loads(antioquia.geometry.to_json()),
                                "below": "traces",
                                "type": "line",
                                "color": "purple",
                                "line": {"width": 1.5},
                                }],})

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show();

"""Es posible continuar añadiendo capas dado que `fig.update_layour` acepta dentro de `layers` una lista de objetos. Estos pueden ser formato GeoPackage, GeoJSON o ShapeFile.

### 3.1.1. Otros recursos

https://geopandas.org/en/stable/getting_started/introduction.html
https://geopandas.org/en/stable/gallery/plotting_basemap_background.html
https://developers.arcgis.com/python/guide/using-the-gis/
https://plotly.com/python/scattermapbox/
https://matplotlib.org/stable/tutorials/index.html

## 3.2. Análisis de regresión lineal

La regresión lineal es uno de los métodos estadísticos más populares. Aunque puede utilizarse para
predecir situaciones no observadas apartir de información conocida (como en el caso del pronóstico de una serie de tiempo o para predecir la situación de un individuo nuevo), su fortaleza radica en las posibilidades que ofrece para la descripción de situaciones, haciendo énfasis en la asociación e incluso causalidad entre variables.

De manera más detallada, mediante el modelo de regresión se pueden resolver preguntas como las siguientes:

* Existe una relacion entre un grupo de variables?
* Si existe, qué tan fuerte es esta relación?
* Qué tan preciso es el efecto calculado de una variable sobre otra?
* Qué tan preciso es el cálculo de valores no observados a partir de información conocida?
* Qué forma toma la relación entre un grupo de variables?

El modelo de regresión lineal (múltiple) se describe como:

<center>$Y = β_0+β_1 X_1+β_2 X_2...+β_p X_p+$ </center>

donde cualquiera de las $X$ y $Y$ son variables aleatorias, $X_j$ representa el j-esimo predictor y $β_j$ es un parámetro poblacional que mide el efecto promedio sobre $Y$ causado por un incremento de una unidad en $X_j$, manteniendo todas las demás variables constantes.

En la práctica, normalmente no se cuenta con toda la información de la $población^1$ sino únicamente con una muestra de individuos que se espera sea representativa.

> $^1$Con la tendencia reciente de proliferación acelerada de información (fenómeno del Big Data) es cada vez más factible pensar en que se dispone de toda la información. Sin embargo, en la práctica, es muy difícil contar con datos para cada uno de los individuos de interés, con lo cual se siguen manejando muestras, eso sí, cada más grandes

En este caso, los valores poblacionales de $β$ no se conocen y se deben reemplazar por estimaciones (muestrales) $\hatβ$, con lo cual se obtiene la ecuación de regresión lineal múltiple:

<center>$Y = \hat{β_0}+\hat{β_1} X_{i1}+\hat{β_2} X_{i2}+...+\hat{β_p} X_{ip} +\hat{\epsilon_i}  $</center>

donde cualquiera de las $X$ y $Y$ son realizaciones de variables aleatorias, $X_{ij}$ representa la realización del j-esimo predictor para el individuo $i$ y $\hat{β_j}$ es una estimación del verdadero parámetro poblacional $β_j$. Para llegar a $\hat{β_j}$ se resuelve el problema de minimización de los errores (poblacionales) al cuadrado:

<center>$\min_{β} {\epsilon^2_1}+{\epsilon^2_2}+...+{\epsilon^2_N} $</center>

donde $\epsilon_i = Y_i - β_0 - β_1 X_{i1} - β_2 X_{i2} - ... - β_p X_{ip}$ Resolviendo el problema de optimización mediante cálculo matricial se encuentra:

<center>$β = (X'X)^{-1}X'Y$</center>

Los coeficientes estimados tienen un error asociado que se relaciona con su precisión: entre mayor sea el error de estimación, menor sería la precisión del coeficiente, y viceversa. Así mismo, este error se puede utilizar para generar estadísticas que permiten validar hipótesis que se tengan sobre los coeficientes de regresión. En este caso, la hipótesis más común es $H_0 : β_j = 0$ versus $H_a : β_j \neq 0$ que, en otras palabras, prueba si la variale $X_j$ tiene una relación *estadísticamente significativa* con la variable $Y$ . Adicionalmente, para que la
estimación de $β$ sea válida, se requiere que $E(X_ϵ) = 0$.

A partir de la información sobre hurto de celulares en Colombia, se utilizará la regresión lineal para establecer el efecto que tienen diferentes variables socioeconómicas sobre este indicador. Una vez más, se carga el archivo Delito.csv mediante el comando:
"""

delito = pd.read_csv('/content/cursotomadecisiones/datos/tema_3/Delito.csv', sep = ';', decimal = ',', encoding='latin-1')
delito.columns

"""El comando columns extrae los nombres de las variables que contiene el objecto delito. Para nuestro
caso, utilizaremos el producto interno bruto (PIB) y una medida de escolaridad (Alumnos) como posibles determinantes de los hurtos de celulares en Colombia. En particular, el PIB viene dado en miles de millones de pesos mientras que el indicador educativo se refiere a la cantidad de estudiantes activos. El diagrama de dispersión entre las variables se obtiene mediante:
"""

import seaborn as sns
plt.style.use('ggplot')

columnas_a_graficar = ['PIB', 'Alumnos', 'Policias']
titles = ['PIB','Estudiantes activos', 'Personal Policía']

fig, axs = plt.subplots(1, 3, figsize=(25,7))
for col, ax, title in zip(columnas_a_graficar, axs.flatten(), titles):
    b = sns.scatterplot(y = 'Celulares', x = col, data = delito, ax = ax, s=40, color='black')
    b.axes.set_title(f'{title}', fontsize=20)

"""Un primer intento por explicar el hurto de celulares como funcion de la producción econòmica y la educación, mediante un modelo de regresión lineal múltiple, es:"""

import statsmodels.api as sm
import statsmodels.formula.api as smf

regresion = smf.ols('Celulares ~ PIB + Alumnos + Policias', data=delito).fit()
regresion.params

"""El resultado obtenido es, como mínimo, escaso, pero esto no signifíca que no se hayan calculado otras estadísticas como valores t calculados, valores p, coeficientes de correlación $R$ y de determinación $R^2$, etc.

Para acceder a una salida mucho más informativa, se utiliza el comando summary:
"""

print(regresion.summary())

pd.DataFrame(regresion.resid.describe()).T

"""En este caso, el hurto de celulares solo muestra una relación positiva, estadísticamente significativa, con el PIB, como se extrae de los valores $p$ calculados para cada coeficiente. Ahora, del mismo ejercicio pero con tasas por cada 100.000 habitantes y producción por persona resulta:"""

delito['cel100'] = delito['Celulares']/delito['Poblacion']*100_000
delito['pib_pc'] = delito['PIB']/delito['Poblacion']
delito['pib_pc_2'] = np.square(delito['pib_pc'])
delito['alum100'] = delito['Alumnos']/delito['Poblacion']*100_000
delito['alum100_2'] = np.square(delito['alum100'])
delito['pol100'] = delito['Policias']/delito['Poblacion']*100_000
delito['pol100_2'] =  np.square(delito['pol100'])

columnas_a_graficar = ['pib_pc', 'alum100', 'pol100']
titles = ['PIB pc', 'Tasa estudiantes activos', 'Tasa personal policía']

fig, axs = plt.subplots(1, 3, figsize=(25,7))
for col, ax, title in zip(columnas_a_graficar, axs.flatten(), titles):
    b = sns.scatterplot(y = 'cel100', x = col, data = delito, ax = ax, s=40, color='black')
    b.axes.set_title(f'{title}', fontsize=20)

regresion = smf.ols('cel100 ~ pib_pc + alum100 + pol100', data=delito).fit()
print(regresion.summary())

pd.DataFrame(regresion.resid.describe()).T

"""En este caso, la transformación de las variables de niveles absolutos a tasas y valores por persona revela que solo hay una relación lineal entre el hurto de celulares y el PIB. Ahora, al observar los diagrama de dispersión correspondientes, se pueden extraer ciertas relaciones no lineales entre las tasa de hurto de celulares y las demás variables explicativas, justifícando el ajuste un modelo con términos no lineales:"""

regresion = smf.ols('cel100 ~ pib_pc + pib_pc_2 + alum100 + alum100_2 + pol100 + pol100_2', data=delito).fit()
print(regresion.summary())

pd.DataFrame(regresion.resid.describe()).T

"""La situación empeora. En este caso, dos de los coeficientes asociados a los indicadores lineales tienen signo positivo, mientras que los coeficiente correspondientes estimados para las variables al cuadrado tienen signo negativo trazando, en suconjunto, parábolas como se ve a continuación para el caso del PIB:"""

fig, axs = plt.subplots(1, 1, figsize=(10,7))
b = sns.scatterplot(y = 'cel100', x = 'pib_pc', data = delito, ax = axs, s=100, color='black')
b.axes.set_title('Tasa hurto de celulares vs PIB pc', fontsize=20)
seq = np.arange(0.001,0.035,0.035/delito.shape[0])
axs.plot(seq, seq*regresion.params[1] + np.square(seq)*regresion.params[2] , color = 'orange');

"""### 3.3. Análisis de regresión espacial

Con la estimación de la ecuación de regresión no sólo se obtienen coeficientes sino también $residuales^1$, definidos como $y_i - \hat{y_i}$, con $\hat{y_i} = \hat{β_0} + \hat{β_1}X_{1i} + \hat{β_2}X_{2i}+ ... + \hat{β_p}X_{pi}$. Estos residuales pueden representarse espacialmente, obteniendo:

> 1: Este es el nombre que se le da a la estimación de los errores de la regresión.
"""

delito['residuales'] = regresion.resid
mapa = gpd.read_file("/content/cursotomadecisiones/datos/tema_3/Colombia/COLOMBIA.shp")
mapa = mapa.merge(delito, on='OBJECTID')

mapa.plot(column='residuales', legend = True, legend_kwds = {'shrink': 0.3},
          figsize = (10,10), cmap = 'YlOrRd', linewidth= 1.5, edgecolor='white');

"""Ahora, si se observa algún patrón geográfico en la distribución de los residuales es indicio de la necesidad de un modelo que capture la dinámica espacial de las observaciones, como en la regresión espacial.

Para finalizar, hablaremos de la autocorrelación espacial, y la definiremos como una medida de qué tan parecidos son dos individuos en función de qué tan cerca (en el espacio) se encuentran. Para su medición debemos contar con un objeto de referencia espacial (mapa) y una medida de cercanía como, por ejemplo, la contigüidad:
"""

from libpysal.weights import Rook

mapa = mapa.set_crs('epsg:4326', inplace= True)
mapa = mapa.to_crs(epsg=3857)

w_rook = Rook.from_dataframe(mapa)

total_len_w = [len(val) for val in w_rook.neighbor_offsets.values()]
min_w = np.min(total_len_w[total_len_w != 0])
max_w = np.max(total_len_w)

print(f'Número de regiones: {w_rook.n}')
print(f'Máximo número de vecinos: {w_rook.max_neighbors}')
print(f'Mínimo número de vecinos: {w_rook.min_neighbors}')
print(f'Número de vecinos promedio: {w_rook.mean_neighbors}')
print(f'Número de links diferentes de 0: {w_rook.nonzero}')
print(f'Porcentaje de links no 0: {w_rook.pct_nonzero}')
print(f'ID de las islas {w_rook.islands}')

print(f'\nRegiones menos conectadas con un mínimo de {min_w} conexiones:')
for k, val in w_rook.neighbor_offsets.items():
    if len(val) == min_w:
        print(k)

print(f'\nRegiones más conectadas con un máximo de {max_w} conexiones:')
for k, val in w_rook.neighbor_offsets.items():
    if len(val) == max_w:
        print(k)

"""En este caso, el comando Rook toma un objeto espacial de polígonos y lo convierte en un objeto que indica cuál departamento limita con cuál(es) departamento(s). 

Visualmente se tiene:
"""

ax = mapa.plot(edgecolor='grey', facecolor='w', figsize = (12,12))
f,ax = w_rook.plot(mapa, ax=ax,
        edge_kws=dict(color='r', linestyle=':', linewidth=1),
        node_kws=dict(marker='o'))
ax.set_axis_off();

"""A continuación se transforma $w$ en una matriz de pesos espaciales, la cual refleja la intensidad de las relaciones espaciales entre los individuos:"""

Wmatrix, ids = w_rook.full()
pd.DataFrame(Wmatrix).head()

"""Una medida que se suele utilizar para medir la autocorrelacion espacial es la $I$ de $Moran^1$, que se calcula de la siguiente forma:

> 1: Existen otras medidas como la C de Geary y el semivariograma.

<center>$I = \frac{N}{\sum_{i=1}^{N} (y_i-\overline{y})^2} \frac{\sum_{i=1}^N \sum_{i=1}^N w_{ij} (y_i-\overline{y}) (y_i-\overline{y})}{\sum_{i=1}^N w_{ij}} $</center>
"""

from esda.moran import Moran

mapa['cel100'] = mapa['cel100'].fillna(0)
mapa['residuales'] = mapa['residuales'].fillna(0)
moran = Moran(mapa['cel100'].values, w_rook)

def reporte_moran(moran):
    print(f'Moran I statistic : {moran.I}')
    print(f'Moran Expectation : {moran.EI}')
    print(f'Moran Variance : {moran.VI_rand}')
    print(f'Moran Variance : {moran.p_rand}')

reporte_moran(moran)

moran = Moran(mapa['residuales'].values, w_rook)
reporte_moran(moran)

"""La prueba se realiza por simulación de Monte Carlo y bajo el supuesto de normalidad, cambian los argumentos para obtener un resultado u el otro. En nuestro caso trabajamos con simulación de Monte Carlo (rand). De acuerdo con los resultados, la tasa de hurto de celulares muestra autocorrelación espacial, mientras que los residuales del modelo de regresión lineal no.

### 3.4. Other resources

https://courses.spatialthoughts.com/python-foundation.html

## 4. Redes Complejas

Desde hace algunos años se ha venido popularizando el uso de redes sociales, donde los individuos se identifican por sus características, y se interrelacionan con otros individuos de maneras muy diferentes: mediante mensajes, *likes*, toques, como seguidores, etc. 

En todos estos casos se genera información que diferente a la que se suele manejar en la mayoría de contextos. A esta información la llamaremos relacional o de relaciones. Aquí, es importante considerar que este tipo de datos no es exclusivo de las redes sociales sino que se genera
en la mayoría de las industrias y campos de la ciencia.

En el análisis de redes se tienen dos elementos principales:

* Nodos (o vértices): son individuos que generan relaciones.
* Bordes: son las relaciones generadas por los nodos.

Estos dos tipos de elementos nos permitirán construir nuestra estructura de red. Adicionalmente, a partir de ellos se construirán diferentes estadísticas que aportan información sobre los individuos, sus relaciones y la red como un todo.

En nuestro caso, trabajaremos información bancaria correspondiente a transacciones donde un individuo le transfíere a otro individuo. La información se encuentra en el archivo *Transacciones.csv*. Inicialmente, cargamos el archivo y vemos sus primeras observaciones:
"""

trans = pd.read_csv('/content/cursotomadecisiones/datos/tema_3/Transacciones.csv', sep = ';', decimal = ',', encoding='latin-1')
trans.head()

"""Los campos `IndividuoA` y `IndividuoB` se refieren a las dos contrapartes que intervienen en cada transacción; `Id` es el identificador de transacciones; cuando Tipo es 5 es una transacción de A hacia B, mientras que Tipo igual a 8 es un movimiento en el sentido contrario; `Valor` es el valor de la transacción; `Fecha` es la fecha de la transacción; Oficina hace referencia al lugar donde ocurre la transacción; y `Municipio` es el municipio donde se ubica la oficina. Ahora, algunas estadísticas descriptivas sobre los datos:"""

trans.info()

len(set(trans['IndividuoA'])|set(trans['IndividuoB']))

trans['Valor'].describe()

trans['Oficina'].nunique()

trans['Municipio'].nunique()

"""Utilizaremos el paquete **networkx** para el procesamiento de la información. Aquí, necesitamos reorganizar los datos en dos conjuntos, uno de ellos sobre individuos y el otro sobre sus relaciones. En ambos casos se pueden añadir características adicionales según sea el caso. El elemento de nodos debe tener, como mínimo, una columna con las identificaciones de los individuos. 

Por su parte, el elemento de bordes tiene dos columnas, la primera de ellas con la identificación de la parte y la segunda con la identificación de la contraparte en la relación:
"""

nodos = pd.Series(pd.unique(trans[['IndividuoA','IndividuoB']].values.ravel())).sort_values()

bordes = pd.DataFrame({'origen': pd.concat([trans[(trans['Tipo']==5)]['IndividuoA'], trans[(trans['Tipo']==8)]['IndividuoB']]),
                       'destino': pd.concat([trans[(trans['Tipo']==8)]['IndividuoA'], trans[(trans['Tipo']==5)]['IndividuoB']])})

print(len(nodos) == len(nodos.unique()))

print(bordes.shape[0] == len(set(bordes['origen'])|set(bordes['destino'])))

"""Por su construcción, no hay individuos repetidos en el elemento nodos, contrario a lo que sucede con el elemento bordes. Entonces, se pueden reducir los bordes para que contenga
combinaciones únicas:
"""

bordes.shape[0]

bordes.drop_duplicates().shape[0]

"""El comando `drop_duplicates` hace la tarea pero pierde información que puede ser valiosa sobre la cantidad de combinaciones que se repiten. Ahora, se utiliza el paquete networkx y la función from_pandas_edgelist, que requiere como mínimo la información de bordes:"""

import networkx as nx

G = nx.from_pandas_edgelist(bordes, 'origen', 'destino')

options = {
    'node_size': 20,
     'node_color':'gold', 
     'linewidths': 0.5,
}

fig = plt.figure(figsize=(17, 17))
ax = plt.subplot(111)
nx.draw(G, **options)

"""En este caso se uso la función `draw` que genera de manera automática la ubicación de cada nodo intentandolo separar de los grupos con los que no tiene relación. Adicionalmente, se pueden utilizar las funciones `draw_circle`, `draw_kamada_kawai`, `draw_planar` y [otros](https://networkx.org/documentation/stable/reference/drawing.html)

Las funciones `edges` y `nodes` generan una lista de los bordes y nodos, respectivamente:
"""

G.edges

G.nodes

"""La información en estructura de red no solo sirve para facilitar la visualización de los individuos y sus relaciones, sino que también permite calcular indicadores de conectividad. Uno de los más básicos es el grado, que corresponde a la cantidad de conexiones que tiene un nodo. Para calcular este indicador utilizaremos la función `degree`:"""

pd.DataFrame(G.degree(), columns=['Individuo','NConex']).sort_values('NConex', ascending=False)

"""## 5. Imagenes como fuente de datos

### 5.1. Crear una base de datos

El procesamiento de imágenes está emergiendo ahora como un espacio novedoso e innovador en la investigación y aplicaciones informáticas.

El procesamiento de imágenes toma imágenes como entrada y las técnicas de procesamiento de imágenes se utilizan para procesar las imágenes y la salida son imágenes modificadas, video o colección de texto o características de las imágenes. La salida resultante de la mayoría de las técnicas de procesamiento de imágenes crea una gran cantidad de datos. En esta técnica, la información voluminosa se procesa y almacena como datos estructurados o no estructurados como resultado del procesamiento de imágenes mediante técnicas informáticas.

Para este ejemplo vamos a crear un base de datos estructurada a partir de imágenes, utilizando las propiedades de las imágenes como un insumo y técnicas de aprendizaje profundo particularmente un modelo denominado [**YOLOv5**](https://es.wikipedia.org/wiki/Algoritmo_You_Only_Look_Once_(YOLO)).

Usualmente, el almacenamiento de imágenes en bases de datos estructuradas (SQL) o semi-estructuradas (JSON) en un servidor local o de nube, guarda exclusivamente el enlace donde están ubicadas. Pero es posible que solo se tenga una ruta u ubicación y no haya almacenamiento alguno. Por lo que se recorrerá una carpeta (igualmente se puede realizar un proceso similar para servicio en Nube como AWS S3 o GCP Cloud Storage) para identificar la ruta de cada imagen de la siguiente manera:
"""

import os
pd.set_option('display.max_columns', 500)

path = './cursotomadecisiones/datos/tema_3/Imagenes/'
imagenes = pd.DataFrame(dict(ruta = [path+file for file in os.listdir(path)]))
imagenes.head()

"""**¿Por qué es importante crear esta base de datos?** 

Se puede implementar soluciones en frameworks como [Tensorflow](https://www.tensorflow.org/?hl=es-419)/[Pytorch](https://pytorch.org/) que se apalancan de bases de datos estructuradas (o esquemas de carpetas con estructuras) para entrenar de manera sencilla modelos de aprendizaje profundo que en conjunto con información adicional se puede generar modelos para solucionar problemas mas complejos. 

Ejemplo: https://arxiv.org/pdf/1612.05251.pdf 

<img src="https://ichi.pro/assets/images/max/724/1*EI-pvzeYlE3Hdrjk9G9rjg.png"
     alt="Ejempplo spotify"
     style="float: left; margin-right: 10px;" />

### 5.2.Quitando duplicados

De las primeras cosas que se debe hacer cuando se tiene un proyecto de Analítica es asegurarse de la calidad de los datos, si nuestros datos son erróneos o no son relevantes los análisis y predicciones que se hagan a partir de los mismos pueden estar incorrectos. 

Cuando trabajamos con imágenes puede pasar que tengamos duplicados con nombres distintos, por eso nos apalancamos en Hash de imagen para identificar los duplicados. Un hash de imagen surge a partir de la idea de los hashes criptográficos.

Un hash criptográfico asigna un archivo de longitud arbitraria a un vector único, donde cambiar un solo bit en el archivo de entrada da como resultado un nuevo vector arbitrario como se muestra a continuación.
"""

# Ejemplo de hashes criptograficos
print(f'hash of A: {hash("aaaaa")}, hash B: {hash("aaaab")}')

"""Los hash de imagen, por otro lado, tienen como objetivo generar vectores similares para imágenes de cuyos bits son similares. El hash de la imagen genera un hash de 256 bits, donde la similitud se puede calcular simplemente como el porcentaje de bits idénticos en los vectores de dos imágenes.

El hash perceptual se utiliza para generar la huella digital de imágenes. El sitio web [pHash](https://github.com/jenssegers/imagehash/blob/master/README.md) explica una vez más la diferencia entre hashes criptográficos y de imagen:

"Un hash perceptual es una huella digital de un archivo multimedia derivada de varias características de su contenido. A diferencia de las funciones hash criptográficas que se basan en el efecto de avalancha de pequeños cambios en la entrada que conducen a cambios drásticos en la salida, los hash perceptuales están "cerca" entre sí, si las características son similares."

Usaremos la libreria `imagehash` y particularmente la funcion `phash` para generar los hash perceptuales de las imagenes que tenemos. Esta función recibe como parametro un archivo de imagen, por lo que cargaremos las imagenes usando la función `Image` de `PIL` (Python Imaging Library).
"""

from tqdm.notebook import tqdm
from PIL import Image
import imagehash
tqdm.pandas()

def get_hash(file_path):
    img = Image.open(file_path)
    img_hash = imagehash.phash(img)
    
    return img_hash.hash.reshape(-1).astype(np.uint8)
    
imagenes['phash'] = imagenes['ruta'].progress_apply(get_hash)

imagenes.head()

"""Una vez tenemos nuestro vector de bits podemos comparar cada imagen contra el resto para saber si existen similitudes de al menos 90% para identificar dos imagenes como duplicadas. """

import imageio

def encontrar_imagenes_similares(threshold = 0.9):
    import itertools

    duplicados = 0
    duplicados_idxs = set()

    for a, b in itertools.combinations(range(len(imagenes)), 2):
        similarity = (imagenes.loc[a,'phash'] == imagenes.loc[b,'phash']).mean()
        if similarity > threshold and not(duplicados_idxs.intersection([a, b])): 
            duplicados_idxs.update([b])
            # Get DataFrame rows
            imagen_1 = imagenes.loc[a,'ruta']
            imagen_2 = imagenes.loc[b,'ruta']
            # Plot Duplicate Images
            duplicados += 1
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,5))
            ax[0].grid(False)
            ax[0].imshow(imageio.imread(imagen_1))
            ax[0].set_title(f'Idx: {a}')
            ax[1].grid(False)
            ax[1].imshow(imageio.imread(imagen_2))
            ax[1].set_title(f'Idx: {b}')
            plt.suptitle(f'{duplicados} | PHASH Similarity: {similarity:.3f}')
            plt.show()
            
    return duplicados_idxs

fila_duplicadas = encontrar_imagenes_similares()

print(f'Encontró {len(fila_duplicadas)} Imagenes duplicadas')
# Quitar duplicados
imagenes = imagenes.drop(fila_duplicadas).reset_index(drop=True)

"""### 5.2. Análisis de datos - Imagenes

Una de las formas en las que se puede entender las imagenes con las que se está trabajando es visualizar ejemplos de manera aleatoria. 

"""

### plotear batch de imagenes
def mostrar_batch_aleatorio_df(df, rows=2, cols=4):
    df = df.copy()
    df = df.sample(rows * cols).reset_index()
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*4, rows*4))
    for i, ax in enumerate(axes.flatten()):
        idx = df.loc[i, 'index']
        img = imageio.imread(df.loc[i, 'ruta'])
        ax.imshow(img)
        ax.grid(False)
        ax.set_title(f'Índice: {idx}')

mostrar_batch_aleatorio_df(imagenes, rows=2, cols=3)

"""Como se mencionó en la introducción es posible agregar nuevas variables a partir de las propiedades de las imagenes como el alto, ancho, número de canales, tipo de canal, el máximo valor por cada canal, el mínimo valor por cada canal, etc.

En este caso usaremos la libreria `ìmageio` que distinto a `PIL` no carga un elemento tipo imagen, sino que crea vectores de bits con los valores por cada canal (en este caso RGB).
"""

image = imageio.imread('./cursotomadecisiones/datos/tema_3/Imagenes/0e89d7ef7392559c216d1689b82c6898.jpg')
image[:3]

"""Una vez leemos la imagen como un vector de bits podemos crear las nuevas variables de la siguiente forma: """

anchos = []
altos = []
rel_aspectos = []
for ruta in tqdm(imagenes['ruta']):
    image = imageio.imread(ruta)
    h, w, _ = image.shape
    altos.append(h)
    anchos.append(w)
    rel_aspectos.append(w / h)


imagenes['alto'] = anchos
imagenes['ancho'] = altos
imagenes['relacion_aspecto'] = rel_aspectos

imagenes.head()

"""### 5.3. Creación de variables a partir de YoloV5

YOLOV5 es la quinta iteración de la familia de detección de objetos Yo Only Look Once, este algoritmo está disponible gratuitamente, es fácil de usar y tiene puntuaciones bastante altas en los puntos de referencia.

Mediante la detección de objetos, las imágenes se pueden clasificar como gatos o perros, se pueden determinar los contornos de las mascotas y se puede contar la cantidad de mascotas en las imágenes.

Esta detección de objetos puede ser una fuente de características y una herramienta fundamental para el preprocesamiento y creación de nuevas variables.

Los algoritmos YOLO tienen el siguiente proceso:
* Cambiar el tamaño a 448 x 448.
* Ejecutar una única red convolucional en la imagen, el objetivo de esta red va a ser obtener una clase de objeto  y el cuadro delimitador que especifica la ubicación del objeto. Cada cuadro delimitador cuenta mínimo con:
    1. Centro del cuadro delimitador (bxby)
    2. Ancho (bw)
    3. Alto (bh)
    4. Valor cis correspondiente a la clase de un objeto (e.j. perro, gato, humano, etc.)

<img src="https://appsilon.com/wp-content/uploads/2018/08/bbox-1.png"
     alt="cuadro delimitador"
     style="float: left; margin-right: 8px;"
     width="500" />

> Para el cálculo de los cuadros delimitadores se divide en una grilla $SxS$ y se intenta obtener un número $N$ de objetos dentro de cada elemento de la grilla. 

<img src="https://appsilon.com/wp-content/uploads/2018/08/yolo-1.png"
     alt="Calculo del cuadro"
     style="float: left; margin-right: 8px;"
     width="500" />

* Finalmente se obtienen los umbrales de las detecciones resultantes del modelo y se dejan aquellos que maximizan la probabilidad de tener un elemento dentro. 

<img src="https://appsilon.com/wp-content/uploads/2018/08/nonmax-1.png"
     alt="valor final"
     style="float: left; margin-right: 8px;"
     width="500" />


Para implementarlo en nuestro ejemplo lo primero que debemos hacer es obtener el modelo de la siguiente forma:




"""

import torch
yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

"""Una vez obtenemos el modelo ya pre-entrenado, creamos una función para obtener el número de mascotas, las clases, las coordenadas de las cajas de cada clase detectada y los valores $X$ y $Y$ mínimos y máximos y por último los pixeles en donde fueron detectadas mascotas (para calcular la proporción de mascotas en cada imagen).


"""

def crear_variables_yolo(file_path, plot=False):
    import matplotlib.patches as patches

    # Leer la imagen
    imagen = imageio.imread(file_path)
    h, w, c = imagen.shape
    
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(8,8))
        ax[0].set_title('Animales detectados en la \n imagen', size=16)
        ax[0].imshow(imagen)
        ax[0].grid(False)
        
    # Obtener los resultados de  YOLOV5 usando Test Time Augmentation para mejores resultados
    resultados = yolov5(imagen, augment=True)
    
    # Crear mascara para pixeles que contienen mascotas, inicializando el set en 0
    mascotas_pixels = np.zeros(shape=[h, w], dtype=np.uint8)
    
    # Diccionario para salvar la información de la imagen
    imagen_info = { 
        'n_mascotas': 0, # Número de mascotas en la imagen
        'labels': [], # Labels de los objectos detectados
        'thresholds': [], # Probilidad score
        'coords': [], # Coordenadas de las cajas
        'x_min': 0, # coordenada minima en x coordinate de la caja
        'x_max': w - 1, # coordenada maxima en x coordinate de la caja
        'y_min': 0, # coordenada minima en y coordinate de la caja
        'y_max': h - 1, # coordenada minima en y coordinate de la caja
    }
    
    # Lista guardar cajas a dibujar 
    mascotas_encontradas = []
    
    # Guardar la información para cada mascota
    for x1, y1, x2, y2, treshold, label in resultados.xyxy[0].cpu().detach().numpy():
        label = resultados.names[int(label)]
        if label in ['cat', 'dog']:
            imagen_info['n_mascotas'] += 1
            imagen_info['labels'].append(label)
            imagen_info['thresholds'].append(treshold)
            imagen_info['coords'].append(tuple([x1, y1, x2, y2]))
            imagen_info['x_min'] = max(x1, imagen_info['x_min'])
            imagen_info['x_max'] = min(x2, imagen_info['x_max'])
            imagen_info['y_min'] = max(y1, imagen_info['y_min'])
            imagen_info['y_max'] = min(y2, imagen_info['y_max'])
            
            # Poner los pixeles de la imagen donde hay una mascota en 1
            mascotas_pixels[int(y1):int(y2), int(x1):int(x2)] = 1
            
            # Agregar la mascota a la lista
            mascotas_encontradas.append([x1, x2, y1, y2, label])

    if plot:
        for x1, x2, y1, y2, label in mascotas_encontradas:
            c = 'red' if label == 'dog' else 'blue'
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=c, facecolor='none')
            ax[0].add_patch(rect)
            ax[0].text(max(25, (x2+x1)/2), max(25, y1-h*0.02), label, c=c, ha='center', size=14)
                
    # Agregar variable de ratio mascota
    imagen_info['mascota_radio'] = mascotas_pixels.sum() / (h*w)

    if plot:
        # Mostrar los pixeles donde hay una mascota
        ax[1].set_title('Pixels que contienen al \n menos una mascota', size=16)
        ax[1].imshow(mascotas_pixels)
        ax[1].grid(False)
        plt.show()
        
    return imagen_info

"""Con la función anterior podemos obtener toda la información necesaria, aquí podemos ver ejemplos de lo realizado:"""

for ruta in imagenes['ruta'].sample(3):
    crear_variables_yolo(ruta, plot=True)

"""Finalmente, se hace una función que recorre todo el conjunto de datos calculando las variables.

Dado que algunas imágenes pueden tener múltiples mascotas sobre los mismos pixeles (cuando puede que sea un gato o un perro), se toma aquella clase que tenga un mayor % de probabilidad.
"""

def crear_variables_yolo_df(ruta_imagenes = imagenes['ruta']):
    # Inicializar un diccionario vacio
    info_imagenes = {
        'n_mascotas': [],
        'label': [],
        'coords': [],
        'x_min': [],
        'x_max': [],
        'y_min': [],
        'y_max': [],
        'mascota_radio': [],
    }

    for idx, ruta in enumerate(tqdm(ruta_imagenes)):
        imagen_info = crear_variables_yolo(ruta, plot=False)
        
        info_imagenes['n_mascotas'].append(imagen_info['n_mascotas'])
        info_imagenes['coords'].append(imagen_info['coords'])
        info_imagenes['x_min'].append(imagen_info['x_min'])
        info_imagenes['x_max'].append(imagen_info['x_max'])
        info_imagenes['y_min'].append(imagen_info['y_min'])
        info_imagenes['y_max'].append(imagen_info['y_max'])
        info_imagenes['mascota_radio'].append(imagen_info['mascota_radio'])
        
        # No todas las imagenes están correctamente clasificadas
        labels = imagen_info['labels']
        if len(set(labels)) == 1: # 1 sola mascota en la imagen
            info_imagenes['label'].append(labels[0])
        elif len(set(labels)) > 1: # Multiples mascotas - obtener la más confiable
            info_imagenes['label'].append(labels[0])
        else: # desconocido, YoLo no encontró ninguna mascota
            info_imagenes['label'].append('desconocido')
    
    return info_imagenes

info_imagenes = crear_variables_yolo_df(imagenes['ruta'])

# Agregar nuevas variables al dataframe
for k, v in info_imagenes.items():
    imagenes[k] = v


pd.set_option('display.max_columns', None)
imagenes.head()

"""El resultado final permite realizar análisis de datos más profundos del conjunto de imágenes y/o crear modelos que se alimenten de esta información adicional. Por dar un ejemplo, se podría explorar las imágenes donde no se encontró al menos una mascota, para nuestro set de datos se incluyó una imagen la cuál se debería eliminar del set de datos. A continuación se visualiza:"""

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
 ruta_imagen_0_mascotas = imagenes[imagenes['n_mascotas']==0]['ruta']
 img = imageio.imread(ruta_imagen_0_mascotas.values[0])
 ax.imshow(img)
 ax.grid(False)

"""### 5.4. Otros Recursos

* Paper YoLo: https://paperswithcode.com/paper/you-only-look-once-unified-real-time-object
* Documentación PyTorch: https://pytorch.org/hub/ultralytics_yolov5/
* Documentación TensorFlow Lite: https://www.tensorflow.org/lite/examples/object_detection/overview 
* Documentación TensorFlow: https://www.tensorflow.org/hub/tutorials/object_detection
"""