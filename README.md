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