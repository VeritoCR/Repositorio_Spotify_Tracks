# Inteligencia Musical y Predicción de Popularidad en Spotify Tracks

**Autores:** Gino Andrades, Verónica Cereceda  
**Fecha:** Septiembre de 2026  
**Versión:** 1.0  
**Asignatura:** Machine Learning (MLY1101) — Evaluación Parcial N°1  
---
## 1. Justificación de Herramientas Colaborativas
Para garantizar la reproducibilidad, trazabilidad y control de versiones en el desarrollo del proyecto, se seleccionaron dos herramientas colaborativas principales:
* **GitHub:** Utilizado como repositorio (Repositorio_Spotify_Tracks) para el control de versiones del código, documentación estructurada y alojamiento del conjunto de datos, asegurando la sincronización entre los integrantes del equipo.
* **Deepnote:** Entorno interactivo basado en Jupyter Notebooks que permitió la ejecución en tiempo real, trazabilidad de celdas y un entorno de ejecución unificado con dependencias estandarizadas en Python.
---
## 2. Descripción del Problema de Negocio y Objetivos

### 2.1 Problema de Negocio
En plataformas como Spotify se publican miles de canciones todos los días. Sin embargo, la gran mayoría pasa desapercibida o no logra escuchas suficientes, mientras que solo un pequeño grupo se convierte en un éxito. El desafío de este proyecto es saber si es posible anticipar qué tan popular será una canción basándonos únicamente en su inteligencia musical, es decir, en sus características de audio y en sus metadatos. Predecir este valor le sirve a las plataformas y a los creadores de música para saber, antes o recién estrenado un tema, qué nivel de tracción de escuchas puede alcanzar y qué atributos acústicos suelen premiar los oyentes dentro de cada género.
### 2.2 Objetivos del Proyecto
* **Objetivo General:**  
Desarrollar, bajo la metodología CRISP-DM, un proceso que permita diagnosticar la calidad del dataset de canciones de Spotify, comprender qué factores se asocian con su nivel de popularidad, y dejar los datos preparados y transformados para una futura etapa de modelamiento.

* **Objetivos Específicos:**
  * Evaluar la calidad del dataset en sus dimensiones de completitud, unicidad y validez.
  * Realizar un análisis exploratorio univariado, bivariado y multivariado sobre las variables del dataset, identificando patrones y correlaciones entre los             atributos de audio, el contenido explícito, los géneros y la popularidad.
  * Diseñar un proceso de preparación y transformación de datos que filtre las anomalías operativas detectadas (registros corruptos y audios no musicales) y             elimine duplicados, evitando fuga de información hacia una futura etapa de modelamiento.
  * Implementar una normalización contextual intra-género para evaluar el mérito acústico relativo de cada canción frente a su propio estilo musical.
  * Evaluar los sesgos de muestreo derivados de las cuotas fijas por género, y verificar el cumplimiento de las normativas de privacidad al operar sin datos de          identificación personal.
---
### 3. Definición de KPIs
#### 3.1 KPIs de Calidad y Preparación de Datos 
* **Índice Global de Calidad del Catálogo:** Medición porcentual consolidada sobre dimensiones de completitud, unicidad y validez.  
  *Resultado obtenido:* **92.86%** de calidad general en el dataset original.
* **Tasa de Depuración de Anomalías e Inconsistencias:** Filtrado del 100% de los registros operativos inválidos (444 duplicados exactos, 16.641 repeticiones por cuota de género, pistas con duración = 0 o > 15 min, y compás inválido), logrando una matriz final limpia de **89.439 pistas únicas**.
* **Normalización Contextual por Género:** en vez de agrupar los 114 géneros en categorías más amplias, se decidió mantenerlos completos y usarlos para calcular un Z-score propio por género en cada atributo acústico, evitando así perder la especificidad de cada estilo musical.

#### 3.2 KPIs de Inteligencia Musical y Asociación
* **Capacidad de Explicabilidad de la Popularidad:** se midió la correlación entre los atributos ya normalizados y *popularity*, usando solo el conjunto de entrenamiento para no filtrar información del conjunto de prueba. Los atributos con mayor influencia positiva fueron *loudness* (0,074) y tener compás 4/4 (0,069); en el otro extremo, el compás 3/4 (-0,055) fue el que más penalizó.
* **Margen de Error de Estimación Tolerado:** lograr una desviación promedio inferior a **10 puntos** de popularidad en el conjunto de prueba (X_test) al momento de implementar los algoritmos en la siguiente entrega.
---
## 4. Fuentes de Datos y Metodología CRISP-DM
### 4.1 Fuentes de Datos
* **Archivo y descarga:** Se utiliza el archivo **Spotify_Tracks_Dataset.csv** descargado directamente desde el repositorio de GitHub.
* **Cantidad de datos:** El dataset tiene 114.000 filas y 21 columnas en total.
* **Contenido de las columnas:**
  * **Datos de la canción:** identificador único (track_id), nombre de la canción (track_name), artista (artists), álbum (album_name) y género (track_genre).
  * **Popularidad y formato:** la variable que queremos predecir (popularity), la duración en milisegundos (duration_ms) y si la letra es explícita (explicit).
  * **Características del audio:** medidas que calcula Spotify como qué tan bailable es (danceability), la energía (energy), el volumen (loudness), la presencia de voz (speechiness), si es acústica (acousticness), si es instrumental (instrumentalness), si es en vivo (liveness), el ánimo o positivismo (valence), la velocidad (tempo), el tono (key), el modo (mode) y el compás (time_signature).
* **¿Por qué sirve este dataset?:** Porque reúne tanto la popularidad histórica como las características de audio de cada canción sin usar datos privados ni personales de usuarios.
### 4.2 Metodología CRISP-DM
El proyecto se desarrolla bajo el estándar industrial **CRISP-DM**, cubriendo en esta primera entrega las siguientes etapas:
1. **Comprensión del Negocio:** Delimitación del problema enfocado en la predicción de tracción de canciones, definición de alcance y formalización de KPIs.
2. **Comprensión de los Datos:** Auditoría de calidad métrica (92.86%), perfilamiento estadístico, análisis de valores atípicos y correlaciones lineales (Pearson) y de rangos (Spearman).
3. **Preparación de los Datos:** Depuración de anomalías técnicas, eliminación de duplicados por pista para prevenir fuga de datos, normalización contextual de atributos acústicos por género y escalamiento diferenciado mediante Scikit-Learn.
---
## 5. Resumen del Análisis Exploratorio de Datos (EDA)
* **Calidad de los datos:**
  * *Completitud (100%):* El dataset está prácticamente completo, solo 1 fila de 114.000 tenía valores nulos.
  * *Unicidad (78,72%):* Es el punto más crítico. Solo el 78,72% son canciones únicas. Se encontraron 444 duplicados exactos por error de carga y 16.641 temas repetidos con los mismos valores de audio en distintos géneros debido al muestreo fijo de 1.000 canciones por género.
  * *Validez (99,86%):* El 99,92% de los valores negativos en loudness son correctos. Solo se detectaron 164 registros inválidos con duration_ms = 0 o compás time_signature = 0.
* **Tratamiento de valores atípicos:**
  * Se separaron los errores reales o audios no estándar.
  * Se mantuvieron los valores altos de speechiness,instrumentalness y liveness, ya que no son errores sino características naturales de géneros como rap, música clásica o grabaciones en vivo.
* **Relación con la popularidad:**
  * Ninguna característica de audio individual explica por sí sola la popularidad,todas tienen correlación casi nula, entre -0.10 y 0.05.
  * En cambio, el género musical y si la canción es explícita sí marcan una diferencia clara en las reproducciones promedio,por ejemplo, pop-film promedia 59 puntos frente a géneros de nicho que apenas superan los 2 puntos.
 
## 6. Preparación y Transformación de Datos
El procesamiento de los datos se dividió en dos etapas para asegurar la reproducibilidad y evitar fuga de información:
### 6.1 Limpieza inicial 
* **Limpieza de estructura:** se eliminó la columna **Unnamed: 0** por ser solo un índice residual y se descartó la única fila con valores nulos.
* **Control de duplicados:** se conservó únicamente la primera aparición de cada track_id, evitando que la misma canción se repita en diferentes géneros y contamine la futura evaluación.
* **Filtro de duración y compás:** se eliminaron canciones con duración igual a 0 ms, aquellas mayores a 15 minutos, y registros con compás igual a 0, ya que un compás no puede tener cero tiempos.
* **Excepción justificada:** se mantuvieron 157 registros con tempo = 0, ya que el 88% corresponde al género *sleep*, donde la ausencia de un tempo detectable es un rasgo real de este tipo de música ambiental, no un error de carga.
* **Ajuste de tipos:**
  * La variable **explicit** se transformó de booleano a número entero.
  * Las variables **key** y **time_signature** se pasaron a texto para que el modelo no las interprete como escalas matemáticas continuas.
* **Eliminación de columnas de alta cardinalidad:** se descartaron artists,album_name,track_name y track_id, ya que su volumen de categorías únicas no aporta valor directo al modelamiento y podría introducir sesgo de "fama comercial" si el modelo aprendiera a asociar popularidad con nombres específicos en vez de con atributos musicales.

### 6.2 Separación de datos
* Antes de aplicar cualquier escalado o transformación estadística, se separó la variable objetivo **popularity** del resto de características.
* Los datos se dividieron en **80% para entrenamiento (X_train)** y **20% para prueba (X_test)** usando una semilla fija de 42.

### 6.3 Ensamble final
* **Normalización Contextual por Género:** en vez de comparar todas las canciones contra un promedio general del catálogo, se calculó la media y desviación estándar de cada atributo acústico **dentro de cada género por separado**, usando solo los datos de entrenamiento. Así, una canción se evalúa frente a lo esperado para su propio estilo musical, y no frente a géneros con una naturaleza sonora distinta. Una vez calculado, se descarta la columna de género, por lo que el modelo no tiene acceso directo a esa etiqueta.
* **RobustScaler:** aplicado a duration_ms, loudness y tempo, utilizando la mediana y el rango intercuartílico para no verse afectado por los valores extremos.
* **OneHotEncoder:** aplicado a key y time_signature, para convertirlas en variables binarias independientes sin asumir una progresión matemática falsa entre sus valores.
---
## 7. Evaluación de Sesgos, Ética y Privacidad
* **Privacidad de los datos:** el dataset contiene únicamente características acústicas y datos públicos de las canciones de la API de Spotify. No incluye ningún dato personal de usuarios, cumpliendo con los estándares de privacidad.
* **Sesgo de muestreo:** cada uno de los 114 géneros tiene exactamente 1.000 canciones en el archivo original. En el consumo real de streaming esto no es así, el pop y la música urbana concentran muchas más reproducciones que géneros como el jazz o la música clásica. Entrenar con cuotas fijas artificiales puede sobreestimar la presencia de géneros pequeños.
* **Sesgo de popularidad por origen:** se observaron diferencias grandes de popularidad promedio entre géneros globales y géneros locales. Frente a este riesgo se usó la normalización contextual por género, descartando después la etiqueta de género, así el modelo no tiene acceso directo para favorecer un estilo solo por su volumen de datos. Sin embargo, esto no corrige el sesgo de fondo, la audiencia real de un género de nicho sigue siendo menor, y ese techo de mercado no desaparece solo porque una canción destaque dentro de su propio grupo.
* 
* **Canciones repetidas y recopilatorios:** las canciones que aparecen en múltiples álbumes o bandas sonoras pueden duplicar métricas de forma engañosa. Al filtrar por track_id único se evita que estas repeticiones distorsionen los resultados del proyecto.
* **Sesgo por historial de artista:** en una primera versión del proyecto se había creado una variable para marcar álbumes sin reproducciones previas. Se descartó esta idea porque terminaba castigando a artistas nuevos que simplemente no llevan tiempo suficiente en la plataforma, no porque su música sea peor. Un modelo entrenado con esa variable habría aprendido más bien a distinguir quién ya es conocido que a evaluar las características reales de la canción, algo contrario al objetivo del proyecto.
* **Documentación de supuestos y limitaciones:** cada decisión tomada durante la limpieza y transformación de los datos queda documentada en este informe y en el notebook, de manera que cualquier persona que retome el proyecto pueda entender el alcance y las limitaciones de lo ya trabajado, en lugar de tratar el proceso como una "caja negra".

---
---
## 8. Estructura del Repositorio

```text
├── data/
│   └── Spotify_Tracks_Dataset.csv
├── notebooks/
│   └── Spotify_Tracks_Analysis.ipynb
├── Imagenes/
│   ├── Análisis de Atípicos.png
│   ├── Análisis de Correlación de Pearson.png
│   ├── Análisis de Correlación de Spearman.png
│   ├── Distribución de Variable Objetivo.png
│   ├── Porcentaje de Canciones con Popularidad Cero.png
│   ├── Relación entre popularity y Género.png
│   ├── Top 10 Características más Influyentes.png
│   └── Top 10 Generos más Populares.png
└── README.md
```           
