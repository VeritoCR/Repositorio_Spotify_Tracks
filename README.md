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
En plataformas como Spotify se publican miles de canciones todos los días. Sin embargo, la gran mayoría pasa desapercibida o no logra escuchas suficientes, mientras que solo un pequeño grupo se convierte en un éxito. El desafío de este proyecto es saber si es posible anticipar qué tan popular será una canción basándonos únicamente en su inteligencia musical,es decir, en sus características de audio y en sus metadatos.Predecir este valor le sirve a las plataformas y a los creadores de música para saber, antes o recién estrenado un tema, qué nivel de tracción de escuchas puede alcanzar y qué atributos acústicos suelen premiar los oyentes dentro de cada género

### 2.2 Objetivos del Proyecto
* **Objetivo General:**  
  Desarrollar, durante el ciclo de la presente evaluación, un proceso reproducible bajo la metodología CRISP-DM que permita diagnosticar la calidad de 114.000 pistas de Spotify, cuantificar la influencia de sus características acústicas y contextuales en el nivel de popularidad, y generar un conjunto de datos preparado con un 100% de integridad estructural para la posterior predicción de tracción musical.

* **Objetivos Específicos:**
  * Evaluar cuantitativamente la calidad del dataset en tres dimensiones estándar (completitud, unicidad y validez), logrando un índice de calidad superior al 90% previo a la transformación.
  * Realizar un Análisis Exploratorio de Datos univariado, bivariado y multivariado sobre las 21 variables, determinando los patrones y la correlación estadística entre los atributos de audio, el contenido explícito, los géneros y la popularidad.
  * Diseñar un pipeline de preparación y transformación modular que filtre el 100% de las anomalías operativas (registros corruptos y audios no musicales) y elimine duplicados para evitar fuga de información.
  * Reducir la alta dimensionalidad de los 114 géneros a un esquema semántico de 12 macro-familias e incorporar un indicador de discos sin tracción para aislar el sesgo de popularidad nula.
  * Evaluar los sesgos de muestreo derivados de las cuotas fijas por género y verificar el cumplimiento de las normativas de privacidad al operar sin datos de identificación personal.
---
### 3. Definición de KPIs
#### 3.1 KPIs de Calidad y Preparación de Datos 
* **Índice Global de Calidad del Catálogo:** Medición porcentual consolidada sobre dimensiones de completitud, unicidad y validez.  
  *Resultado obtenido:* **92.86%** de calidad general en el dataset original.
* **Tasa de Depuración de Anomalías e Inconsistencias:** Filtrado del 100% de los registros operativos inválidos (444 duplicados exactos, 16.641 repeticiones por cuota de género, pistas con duración = 0 o > 15 min, y tempo = 0), logrando una matriz final limpia de **89.444 pistas únicas**.
* **Eficiencia en Reducción de Dimensionalidad Categórica:** Compactación del espacio de géneros musicales de 114 categorías nominales a **12 macro-géneros semánticos**, reduciendo la dispersión en un **89.5%**.

#### 3.2 KPIs de Inteligencia Musical y Asociación
* **Capacidad de Explicabilidad de la Popularidad:** Nivel de asociación lineal y no lineal identificado entre los atributos procesados y el target popularity. La inclusión del transformador de GhostAlbumDeriver logró aislar el impacto de la popularidad nula con una correlación de **-0.41**.
* **Margen de Error de Estimación Tolerado:** Lograr una desviación promedio inferior a **10 puntos** de popularidad en el conjunto de prueba (`X_test`) al momento de implementar los algoritmos en la siguiente entrega.
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
3. **Preparación de los Datos (Data Preparation):** Depuración de anomalías técnicas, eliminación de duplicados por pista para prevenir fuga de datos, reducción semántica a 12 macro-géneros, ingeniería de variables (GhostAlbumDeriver) y escalamiento diferenciado mediante Scikit-Learn.
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
* **Control de duplicados:** se conservó únicamente la primera aparición de cada `track_id`, evitando que la misma canción se repita en diferentes géneros y contamine la futura evaluación.
* **Filtro de duración:** se eliminaron canciones con duración igual a 0 ms y aquellas mayores a 15 minutos.
* **Ajuste de tipos:**
  * La variable **explicit**  se transformó de booleano a número entero.
  * Las variables **key** y **time_signature** se pasaron a texto para que el modelo no las interprete como escalas matemáticas continuas.
* **Agrupación de géneros:** los 114 subgéneros se redujeron a **12 macro-géneros** (Pop,Rock_Punk_Indie,Metal,Electronic_Dance,HipHop_RnB,Latin_Caribbean,Acoustic_Folk,Classical_Ambient,Jazz_Blues,Entertainment,Mood y World_Regional).

### 6.2 Separación de datos
* Antes de aplicar cualquier escalado o transformación estadística, se separó la variable objetivo **popularity** del resto de características.
* Los datos se dividieron en **80% para entrenamiento (X_train)** y **20% para prueba (X_test)** usando una semilla fija de 42.

### 6.3 Transformador dinámico
* Se construyó una clase personalizada para identificar álbumes sin tracción comercial previa.
* Se ajusta (.fit()) únicamente con los datos de entrenamiento para aprender los álbumes inactivos. Si en los datos de prueba aparece un álbum desconocido, se clasifica como fantasma por precaución. Esta variable (is_ghost_album) demostró una fuerte correlación negativa de **-0.41** con la popularidad.

### 6.4 Ensamble final
* **RobustScaler:** aplicado a duration_ms, loudness y tempo, utilizando la mediana y el rango intercuartílico para no verse afectado por los valores extremos.
* **MinMaxScaler:** aplicado a las métricas acústicas que ya vienen acotadas de fábrica entre 0.0 y 1.0 (anceability,energy,speechiness,acousticness, instrumentalness,liveness,valence), además de mode y explicit.
* **OneHotEncoder:** aplicado a las columnas de texto (macro_genre,key y time_signature) para convertirlas en variables binarias independientes.
---
## 7. Evaluación de Sesgos, Ética y Privacidad
* **Privacidad de los datos :** el dataset contiene únicamente características acústicas y datos públicos de las canciones de la API de Spotify. No incluye ningún dato personal de usuarios, cumpliendo con los estándares de privacidad.
* **Sesgo de muestreo:** cada uno de los 114 géneros tiene exactamente 1.000 canciones en el archivo original.En el consumo real de streaming esto no es así,el pop y la música urbana concentran muchas más reproducciones que géneros como el jazz o la música clásica.Entrenar con cuotas fijas artificiales puede sobreestimar la presencia de géneros pequeños.
* **Sesgo de popularidad por origen:** se observaron diferencias grandes de popularidad promedio entre géneros globales (pop-film con 59 puntos) y géneros locales o tradicionales (iranian con 2 puntos). Un sistema que use esta métrica sin supervisión puede tender a favorecer siempre a los artistas comerciales y dejar fuera a creadores independientes o regionales.
* **Canciones repetidas y recopilatorios:** las canciones que aparecen en múltiples álbumes o bandas sonoras pueden duplicar métricas de forma engañosa.Al filtrar por track_id único y crear la variable de álbum fantasma, se evita que estas repeticiones distorsionen los resultados del proyecto.
---
## 8. Estructura del Repositorio
```text
├── data/
│   └── Spotify_Tracks_Dataset.csv  
├── notebooks/
│   └── Spotify_Tracks_Analysis.ipynb
├── README.md                      
