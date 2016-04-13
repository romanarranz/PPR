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


## Licencia

MIT License

Copyright (c) [2016] [Román Arranz Guerrero]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.