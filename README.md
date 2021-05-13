# Olimpiadas_Predicción
## Predicción de la cantidad de medallas que ganarán los países en las olimpiadas

### Descripción del Dataset y cómo se obtuvo
##### Se utilizaron datos de juegos olímpicos publicados en Kaggle. Éste cuenta con dos archivos de conjuntos de datos en formato CSV y ambos se utilizarán para hacer el análisis. En el archivo athlete_events se encuentra la información de todos los atletas y competencias en las que participaron, así como la información de las medallas ganadas. En noc_regions se encuentra la información del código del Comité Olímpico Nacional y el país correspondiente de cada código. Los datos de cada uno de los dos dataset se unieron de manera estratégica para tomar provecho de ellos. 
##### Dado a que la base de datos de una de las dos tablas llamada athlete_events.csv es bastante grande, fue necesario comprimirlo en un .rar 

### Objetivos
##### Con este proyecto lo que se pretendió hacer, además de poner en práctica habilidades en preprocesamiento de datos y modelos de machine learning para problemas de regresión, fue enfocarlo a la predicción de futuros resultados de un evento deportivo de importancia, y dado a que el mismo principio puede ser aplicado a competencias de distinta índole, este tipo de programas resultaría de gran ayuda a apostadores y analistas deportivos. 
##### Dado a que en los trabajos reales en ciencia de datos los conjuntos de datos suelen estar dispersos en más de una tabla y de manera desorganizada, resulta muy útil trabajar en este tipo de proyectos donde se necesita más de un dataset para hacer nuestro análisis de machine learning. 

#### Herramientas y librerías utilizadas
##### Se utiliza la librería de Pandas para importar el archivo correspondiente y gestionar los datos de manera adecuada antes de pasarlos por el modelo de Machine Learning. Es útil para hacer el análisis de cada una de las variables y eliminar columnas y/o filas con datos nulos, unir datos de distintas tablas, así como para hacer el preprocesamiento correspondiente de los datos y hacerlos más viables pare el modelo predictivo. En específico para este proyecto fue necesario un extenso preprocesamiento de los datos para reducir el número de muestras a las necesitadas por medio de una serie agrupaciones útiles, así como de conversiones de variables literales a numéricas.
##### Se utiliza la librería de Matplotlib y Seaborn para hacer gráficos que permitan ir entendiendo los datos y visualizarlos por medio de histogramas, gráficos de dispersión y gráficos de unión de puntos. Se usa también la librería de Seaborn ya que a veces es mejor la manera en que se presentan aquí que en Matplotlib. 
##### Se utiliza también la librería de scikitlearn para resolver un problema de regresión con un modelo de machine learning. Los datos de entrenamiento serán aquellos antes de las últimas olimpiadas en Rio de Janeiro en 2016, y las de prueba serán las de Rio 2016. En realidad se entrenan tres modelos para que aprenda a predecir tanto las medallas de oro, como las de plata y bronce en las siguientes olimpiadas.  

### Conclusiones y resultados obtenidos
##### En el algoritmo de Regresión Lineal Múltiple utilizado hay un valor de R-cuadrado bastante bueno para la predicción de cada uno de la cantidad de medallas oro, plata y bronce. A continuación los valores respectivos:
![image](https://user-images.githubusercontent.com/43154438/118084027-92cf1800-b385-11eb-9b4e-5dab6646446d.png)
##### Estos son los datos cuando se toman como datos de prueba los de Rio de Janeiro 2016.
##### Podemos ver a continuación una parte de la tabla de datos de prueba (y_test) donde es posible visualizar qué tan aproximados están los datos predichos a los reales.
![image](https://user-images.githubusercontent.com/43154438/118084069-a7131500-b385-11eb-8d60-ad59484c19b2.png)
##### Podemos concluir que la aproximación es considerable en algunos casos y en otros se aleja un poco. Pero cabe resaltar que si se le agregan más datos de relevancia además de los disponibles en esta base de datos, tales como promedio de tiempo de entrenamiento por cada atleta, número de eventos deportivos en el que participaron los atletas, así como el desempeño deportivo de los atletas los últimos años; entonces así podría tenerse un resultado más acercado y un mejor desempeño.
##### Además también se creó un nuevo dataset para simular los resultados reales de los Juegos Olímpicos que supuestamente se iban a llevar a cabo en Tokyo. Cabe destacar que para este habría sido conveniente tomar los datos de Rio 2016 como datos de entrenamiento. 



