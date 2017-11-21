# DocumentClusteringBigData

Por: José Luis Álvarez Herrera y Luis Alfredo Gallego Montoya

# Descripción de la Aplicación

Este proyecto es una implementación en Pyspark de la técnica Clustering
de Machine Learning No Supervisado usando el algoritmo K-Means. Basado en
el paper "Similarity Measures for Text Document Clustering" de Anna Huang
para el Departamento de Ciencias de la Computación de la 
Universidad de Waikato, Hamilton, Nueva Zelanda. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.332.4480&rep=rep1&type=pdf

La aplicación fue realizada para la práctica #4 de la materia Tópicos Especiales
en Telemática de la Universidad EAFIT.

# Implementación

La implementación es en sí un script, que carga los datos en un RDD con parejas (key, value), donde 
las keys son las rutas de ubicación de los textos de la base de datos de Gutenberg y los values corresponden
al documento representado como un String, estos String luego son mapeados por la función Map para remover los
caracteres no ASCII a la vez que los separa por términos separados. 

Posteriormente la clase HashingTF, mapea los términos encontrados en los String de contenido de documento con
sus respectivas frecuencias, para luego usar la clase IDF que es frecuencia inversa de documentos, usada en el
estudio de Anna Huang.

Luego usamos la librería Mllib de Apache Spark, para entrenar el modelo del K-Means. A esto le pasamos el K,
los datos a clusterizar, las iteraciones máximas y la forma de inicialización del algoritmo en este caso "random".

# Ejecución

```
  spark-submit --master yarn --deploy-mode cluster --executor-memory 1G --num-executors 4 MainBigData.py
```
