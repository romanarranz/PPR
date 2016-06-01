Programación Paralela
===================

Prácticas de la asignatura Programación Paralela cursada en el Grado en Informática de la **Universidad de Granada**

----------

Índice de Prácticas
-------------
#### Práctica 1: Implementación distribuida de un algoritmo paralelo de datos usando MPI
En esta práctica se aborda la implementación paralela del algoritmo de **Floyd** para el cálculo de todos los caminos más cortos en un grafo etiquetado.

> **Consideraciones:**

> - Se tiene inicialmente una implementación secuencias y básica del algoritmo de Floyd y una clase Grafo como TDA para almacenar los datos.
> - De forma auxiliar se tiene un ejecutable para crear ejemplos para N vértices indicado como parámetro de entrada.
> - Sólo será necesario realizar las implementaciones paralelas del algoritmo.
> 
> **Implementaciones:**
> 
> - Descomposición unidimensional (por bloques de filas)
> - Descomposición bidimensional (por bloques 2D)

#### Práctica 2: Implementación distribuida de un algoritmo paralelo usando equilibrado de carga
En esta práctica se aborda la implementación paralela del algoritmo del problema conocido TSP ó Viajante de Comercio en el que usaremos una clase Grafo como TDA para almacenar los datos: ciudades (vértices) y distancias entre ciudades (arcos etiquetados).

> **Consideraciones**

> - Para la implementación usaremos técnicas de ramificación y poda (Branch & Bound) en la que dinámicamente construimos un árbol de búsqueda, su raíz sería el problema inicial y los nodos hoja serían los caminos entre ciudades.
> - Para encontra la solución del problema tendremos que usar un sistema que detecte la situación de fin que implementaremos mediante un algoritmo de paso de testigo en anillo (Algoritmo de Terminación de Dijkstra).
> 
> **Única implementación**


#### Práctica 3: Implementación de un algoritmo paralelo usando NVidia CUDA
Durante la elaboración de esta práctica se han desarrollado diferentes implementaciones del algoritmo Floyd Warshall para encontrar los caminos mínimos entre ciudades en la cual se ha usado CUDA como herramienta de programación ofrecida por NVidia.

> **Consideraciones**
> 
> - Se recuerda que los kernels de CUDA se trata del codigo que ejecuta una sola hebra.
> - Se han implementado diferentes formas de abordar el algoritmo:
> 	- Floyd1D recorriendo las filas de la matriz M.
> 	- Floyd2D distribucion por bloques de nHebras * nHebras de la martiz.
> 	- Floyd1DShared recorrido por filas de la matriz M en la cual la filak y la columnak se encuentran en la memoria compartida de la gráfica.
> 	- Floyd2DShared distribución por bloques en los cuales la subfilak y la subcolumnak se encuentran en la memoria compartida de la gráfica.
> - Se debe compilar usando las rutas de **nvcc** propias de su sistema.


## Licencia

Los detalles se encuentran en el archivo `LICENSE`. En resumen, todo el contenido tiene como licencia **MIT License** por lo tanto es totalmente gratuito su uso para proyectos privados o de uso comercial.