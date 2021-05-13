# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 18:11:21 2020

@author: Daniel Andres
"""
### IMPORTAR LIBRERÍAS ###
import pandas as pd
import numpy as np

### IMPORTAR LOS DATOS ###
atletas = pd.read_csv('athlete_events.csv')
paises = pd.read_csv('data/noc_regions.csv')

#Se elimina la columna "notes" del conjunto de datos paises
paises.drop('notes', axis=1, inplace=True)

#Se renombra la columna "region" como "country" por paises
paises.rename(columns = {'region':'Country'}, inplace = True)

#Se unen los conjuntos de datos atletas y paises
data = atletas.merge(paises,left_on = 'NOC', right_on = 'NOC', how = 'left')
data.head()

#Verificamos si existe algún atleta que haya quedado sin país
paises_nulos = data.loc[data['Country'].isnull(),['NOC','Team']].drop_duplicates()

#Se incluye el país en las filas en donde no se encontraba
data['Country'] = np.where(data['NOC']=='SGP', 'Singapore', data['Country'])
data['Country'] = np.where(data['NOC']=='ROT', 'Refugee Olympic Athletes', data['Country'])
data['Country'] = np.where(data['NOC']=='UNK', 'Unknown', data['Country'])
data['Country'] = np.where(data['NOC']=='TUV', 'Tuvalu', data['Country'])

#Se elimina la columna "Team"
data.drop('Team', axis = 1, inplace = True)

#Se renombra la columna "name" como "Athletes"
data.rename(columns = {'Name':'Athletes'}, inplace = True)

tipos_datos = data.dtypes

#Conocer los datos nulos
datos_nulos = data.isnull().sum()

### VISUALIZACIÓN DE LOS DATOS ###
import matplotlib.pyplot as plt 
import seaborn as sns

#Distribución de las medallas de oro por edad
oro = data[(data.Medal == 'Gold')]
plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(oro['Age'])
plt.title('Distribución de Medallas de Oro')

#Disciplinas con más medallas de oro repartidas
disciplina = oro['Sport']
plt.figure(figsize=(20,10))
plt.tight_layout()
ax = plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
sns.countplot(disciplina)
plt.title('Distribución de Medallas')

#Distribucipón de los 5 países con más medallas de oro
totalOro = oro.Country.value_counts().reset_index(name='Medal').head(5)
g = sns.catplot(x="index", y="Medal", data=totalOro,
                height=6, kind="bar", palette="muted") 
g.despine(left=True)
g.set_xlabels("Países")
g.set_ylabels("Número de Medallas")
plt.title('Medallas por País')

plata = data[(data.Medal == 'Silver')]
bronce = data[(data.Medal == 'Bronze')]

# #Distribucipón de los 5 países con más medallas de bronce
# totalBronce = bronce.Country.value_counts().reset_index(name='Medal').head(5)
# g = sns.catplot(x="index", y="Medal", data=totalBronce,
#                 height=6, kind="bar", palette="muted") 
# g.dispine(left=True)
# g.set_xlabels("Países")
# g.set_ylabels("Número de Medallas")
# plt.title('Medallas por País')

### PROCESAMIENTO DE LOS DATOS ###
#Se elimina toda la información anterior al año 1960
data = data.loc[(data['Year'] > 1960), :]

#Se elimina la información de los juegos olímpicos de invierno
data = data.loc[(data['Season'] == 'Summer'), :]

#Cambiar las columnas de medallas por valores numéricos
data['Gold'] = data['Medal'] == 'Gold'
data['Gold'] = data['Gold'] * 1
data['Silver'] = data['Medal'] == 'Silver'
data['Silver'] = data['Silver'] * 1
data['Bronze'] = data['Medal'] == 'Bronze'
data['Bronze'] = data['Bronze'] * 1

data['Female'] = data['Sex'] == 'F'
data['Female'] = data['Female'] * 1

#Se agrupa la información por país de atletas, deportes y eventos particiádos
data_final = pd.DataFrame(data.groupby(['Year', 'NOC', 'Country'])
                          ['Athletes','Sport','Event'].nunique())
data_final.head()

#Se agrupa la información por país de los géneros
genero = pd.DataFrame(data.groupby(['Year','NOC','Country','Athletes'])
                      ['Female'].mean())
genero = genero.groupby(['Year', 'NOC', 'Country']).sum()
data_final = data_final.merge(genero, left_index=True, right_index=True)
data_final.head()

#Se suman las medallas ganadas por país
medallas = pd.DataFrame(data.groupby(['Year','NOC','Country','Event'])
                        ['Medal'].nunique())
medallas = medallas.groupby(['Year','NOC','Country']).sum()
data_final = data_final.merge(medallas, left_index=True, right_index=True)
data_final.head()

#Se suman los tipos de medallas ganadas por país
medallasTipo = data.groupby(['Year','NOC','Country','Event'])['Gold','Silver','Bronze'].sum()
medallasTipo = medallasTipo.clip(upper=1)
medallasTipo = medallasTipo.groupby(['Year','NOC','Country']).sum()
data_final = data_final.merge(medallasTipo, left_index=True, right_index=True)
data_final.head()

#Se resetea la numeración (agregar columna index)
data_final = data_final.reset_index()
data_final.head()

#Se calcula la cantidad de atletas por eventos
data_final['Athletes Event'] = (data_final['Athletes'] / data_final['Event']).round(3)
data_final.head()

#Convertimos la columna NOC en datos numéricos
data_final = pd.get_dummies(data=data_final, columns=['NOC'])
data_final.head()

### ANÁLISIS DE MACHINE LEARNING ###
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#Definir las variables dependiente e independientes
y = data_final[['Medal','Gold','Silver','Bronze','Year','Country']]
X = data_final.drop(['Medal','Gold','Silver','Bronze'], axis = 1)


# f = plt.figure(figsize=(50,40))
# plt.matshow(data_final.corr('pearson'), fignum=f.number)
# plt.xticks(range(data_final.shape[1]), data_final.columns, fontsize=8, rotation=45)
# plt.yticks(range(data_final.shape[1]), data_final.columns, fontsize=10)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Matriz Correlación Pearson')
# plt.show()

#Separar los datos de entrenamiento y prueba
X_train = X[X['Year'] < 2016]
X_test  = X[X['Year'] == 2016]
X_test  = X_test.reset_index().drop(['index'], axis=1)

y_train = y[y['Year'] < 2016]
y_test  = y[y['Year'] ==2016]
y_test  = y_test.reset_index().drop(['index'], axis=1)

y_test = y_test.drop(y_test[y_test['Country']=='Kosovo'].index)
y_test = y_test.drop(y_test[y_test['Country']=='Refugee Olympic Athletes'].index)
y_test = y_test.drop(y_test[y_test['Country']=='South Sudan'].index)

# y_test = y_test.drop(y_test[y_test['Gold Prediction']>100].index)
# y_test = y_test.drop(y_test[y_test['Silver Prediction']>100].index)
# y_test = y_test.drop(y_test[y_test['Bronze Prediction']>100].index)


#Definir el algoritmo
algoritmo_oro = LinearRegression()
algoritmo_plata =LinearRegression()
algoritmo_bronce = LinearRegression()

#Entrenar el algoritmo
algoritmo_oro.fit(X_train.drop(['Country'], axis=1), y_train['Gold'])
algoritmo_plata.fit(X_train.drop(['Country'], axis=1), y_train['Silver'])
algoritmo_bronce.fit(X_train.drop(['Country'], axis=1), y_train['Bronze'])

#Realizar una predicción con los datos de los Juegos Olímpicos de Rio 2016
y_test['Gold Prediction'] = pd.DataFrame(
    algoritmo_oro.predict(X_test.drop(['Country'], axis=1)),
    columns=['Gold Prediction'])
y_test['Gold Prediction'] = y_test['Gold Prediction'].astype('int64')
y_test['Gold Prediction'] = y_test['Gold Prediction'].clip(lower=0)

y_test['Silver Prediction'] = pd.DataFrame(
    algoritmo_plata.predict(X_test.drop(['Country'], axis=1)),
    columns=['Silver Prediction'])
y_test['Silver Prediction'] = y_test['Silver Prediction'].astype('int64')
y_test['Silver Prediction'] = y_test['Silver Prediction'].clip(lower=0)

y_test['Bronze Prediction'] = pd.DataFrame(
    algoritmo_bronce.predict(X_test.drop(['Country'], axis=1)),
    columns=['Bronze Prediction'])
y_test['Bronze Prediction'] = y_test['Bronze Prediction'].astype('int64')
y_test['Bronze Prediction'] = y_test['Bronze Prediction'].clip(lower=0)

y_test['Medal Prediction'] = (y_test['Gold Prediction']
                              + y_test['Silver Prediction']
                              + y_test['Bronze Prediction'])

#Calculo de la precisión del modelo
#Calculo R2

print(r2_score(y_test['Gold'],y_test['Gold Prediction']))
print(r2_score(y_test['Silver'],y_test['Silver Prediction']))
print(r2_score(y_test['Bronze'],y_test['Bronze Prediction']))

### PREDICCIÓN JUEGOS OLÍMPICOS TOKIO 2020 ###
#Calculamos la media de los datos desde el 2008
tokio_2020 = data_final[data_final['Year'] >= 2008]

tokio_2020['Year'] = 2020

tokio_2020 = tokio_2020.groupby(
    ['Year','Country'])['Athletes','Sport','Event','Female'].mean().astype('int64')
tokio_2020 = tokio_2020.reset_index()

#Calculamos los atletas por eventos de acuerdo a la media
tokio_2020['Athletes Event'] = tokio_2020['Athletes'] / tokio_2020['Event']
tokio_2020['Athletes Event'] = (tokio_2020['Athletes Event'].fillna(0)).round(3)

#Unimos los datos con la información de NOC
noc = X_test.drop(['Year','Athletes','Sport','Event','Female','Athletes Event'], axis=1)
tokio_2020 = tokio_2020.merge(noc)

#Definir las variables dependientes e independientes
y_tokio = tokio_2020[['Year','Country']]
X_tokio = tokio_2020.drop(['Country'], axis=1)

#Realizar una predicción con los datos de los Juegos Olímpicos de Tokio 2020
y_tokio['Gold Prediction'] = pd.DataFrame(
    algoritmo_oro.predict(X_tokio), columns=['Gold Prediction'])
y_tokio['Gold Prediction'] = y_tokio['Gold Prediction'].astype('int64')
y_tokio['Gold Prediction'] = y_tokio['Gold Prediction'].clip(lower=0)

y_tokio['Silver Prediction'] = pd.DataFrame(
    algoritmo_plata.predict(X_tokio), columns=['Silver Prediction'])
y_tokio['Silver Prediction'] = y_tokio['Silver Prediction'].astype('int64')
y_tokio['Silver Prediction'] = y_tokio['Silver Prediction'].clip(lower=0)

y_tokio['Bronze Prediction'] = pd.DataFrame(
    algoritmo_bronce.predict(X_tokio), columns=['Bronze Prediction'])
y_tokio['Bronze Prediction'] = y_tokio['Bronze Prediction'].astype('int64')
y_tokio['Bronze Prediction'] = y_tokio['Bronze Prediction'].clip(lower=0)

y_tokio['Medal Prediction'] = (y_tokio['Gold Prediction']
                              + y_tokio['Silver Prediction']
                              + y_tokio['Bronze Prediction'])

