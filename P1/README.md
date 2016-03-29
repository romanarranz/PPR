Memoria de Práctica
===================

Se pide implementar el cálculo de todos los caminos mínimos usando el algoritmo de Floyd y haciendo uso del lenguaje de programación C++ y MPI para las versiones paralelas.

La clase *Graph* se encarga de almacenar el grafo etiquetado.
El programa proporcionado *creaejemplo.cpp*  crea archivos de entrada que quedan almacenados en la carpeta input, para ello se tiene que indicar como parámetro de entrada el número de vértices.

Los datos de entrada que tenemos son los siguientes: 60, 240, 480, 600, 800, 920, 1000, 1200.

----------

##1. Versión 1D. Distribución por bloques

La implementación de esta versión está dentro de la carpeta Floyd-1.
En esta versión realizaremos una distribución por bloques de filas, asumimos que el número de vértices *N* es múltiplo del número de procesos *P*.
Se hace un reparto entre los procesos por bloques contiguos de filas, de esta manera cada proceso almacena *N/P* filas de la matriz.
Se podrán usar *N* procesos. Cada uno de ellos se hace cargo de sus filas correspondientes adyacentes de la matriz, y además ejecuta el siguiente algoritmo:

```c++
for(k = 0 ... N-1)
	broadcast(filak)
	for(i = rank*(N/P) ... (rank+1)*(N/P)-1)
		for(j = 0 ... N-1)
			I[i][j] = min(I[i][j], I[i][k] + I[k][j])
```

 - I[i][j],  tanto la *i* como la *j* son locales
 - I[i][k],  *i* es local pero *k* es remota
 - I[k][j],  *k* es remota pero *j* es local

Cada proceso debe conocer su **fila i** y **columna j** locales.
La **fila k** debe ser conocida por todos los procesos, por lo tanto cada proceso debe mandar su *fila k* a todos a través de un ***Broadcast***,  si se da el caso de que la *fila k* pertenece a su bloque no habría problema, pero en el caso en que esta fila no pertenezca a su bloque es entonces cuando otro proceso debe indicársela.

Para la resolución de esta versión tendremos que hacer un reparto de la matriz mediante un ***Scatter***,  acto seguido durante la ejecución del algoritmo de cada proceso tendremos que hacer un ***Broadcast*** y por último tendremos que reunir todos los resultados obtenidos por cada uno de los procesos nuevamente en la matriz, para ello usaremos ***Gather***.

### 1.1 Problemas encontrados y su resolución
####1.1.1 Indices globales a locales
El problema que se nos presenta fue hallar el indice de partida de la *fila k* ya que en mi implementación no usé matrices sino que usé un **vector** de tamaño ***(nverts\*nverts)/numeroProcesos***, por lo tanto para obtener los indices en cada etapa k tuve que hacer lo siguiente: 
```c++
indicePartidaFilaK = (k*nverts)%tamVectorLocal;
```
Una vez que obtuve el indice desde donde comenzaba la fila k de ese proceso tuve que rellenar el vector de la fila k para poder hacer un *Broadcast* de este. Y para rellenarlo tuve que hacer uso del desplazamiento de i hasta el numero de vértices de esa fila partiendo del indice obtenido.

```c++
for(i = 0; i<nverts; i++)
{
	filak[i] = vectorLocal[indicePartidaFilaK + i];
}
```
####1.1.2 Broadcast fila k
Para hacer el *Broadcast* de la fila k tuve que obtener en primer lugar el proceso que se iba a encargar de realizar la difusión, para ello:

```c++
int tamVectorLocal = (nverts*nverts)/numeroProcesos;
int tamFilaLocal = tamVectorLocal/nverts;
// ..........
for(k = 0; k<nverts; k++)
{
	idProcesoBloqueK = k / tamFilaLocal;
	// ..........
}
```

Una vez que obtuve el proceso que se encargaba de ese bloque procedo a hacer la difusión de la *fila k*.

```c++
MPI_Bcast(&filak[0], nverts, MPI_INT, idProcesoBloqueK, MPI_COMM_WORLD);
```

####1.1.3 Acceso a los datos del vector local
Al comienzo cuando realizaba el acceso a los valores del vector local obtenía errores por violación de segmento y esto se debía a que estaba accediendo a indices superiores que no estaban definidos en el propio vector.
Para solucionarlo hice operaciones modulo tamaño del vector.
```c++
for(i = iLocalInicio; i<iLocalFinal; i++)
{
	for(j = 0; j<nverts; j++)
    {
	    // no iterar sobre la diagonal de la matriz
	    if (i!=j && i!=k && j!=k) 
        {   
	        vikj = vectorLocal[(i*nverts)%tamVectorLocal + k] + filak[j];
            vikj = min(vikj, vectorLocal[(i * nverts)%tamVectorLocal + j]);
            vectorLocal[(i*nverts)%tamVectorLocal + j] = vikj;
         }
	}
}
```

Seguramente con una implementación con matrices no hubiera tenido este tipo de problemas, por ello en la versión 2D usé matrices.

##2. Versión 2D. Distribución por submatrices
La implementación de esta versión está dentro de la carpeta Floyd-2.
En esta versión realizaremos una distribución por bloques bidimensionales o 2D, asumimos que el número de vértices *N* es múltiplo de la raíz del número de procesos *P*.

Suponemos que los procesos se organizan lógicamente formando una malla cuadrada con ***sqrtP*** procesos en cada fila y columna. De tal forma que tendremos un bloque por cada proceso de ***N/sqrtP*** filas y cada una de ellas con ***N/sqrtP*** elementos.


En cada etapa k del algoritmo los procesos necesitan saber ***N/sqrtP*** valores de la ***fila k*** y ***columna k***, estos valores estarán dentro del bloque de otros dos procesos, la *fila k* dentro de uno y la *columa k* dentro de otro. Para que los procesos conozcan la *columna k* y *fila k* tendrán que realizarse dos ***Broadcast*** :

 - La columna k será repartida al resto de procesos de la misma fila de la malla de procesos.
 - La fila k será repartida al resto de procesos de la misma columna de la malla de procesos.

Dentro del guión de la práctica se facilitaba el código para realizar el **empaquetado** de los datos. Este proceso solo podía ser realizado por **P0** y de este proceso obteníamos como resultado un buffer de salida relleno con los datos de las submatrices de cada proceso.

### 2.1 Problemas encontrados y su resolución
#### 2.1.1 Comunicadores
Para completar la tarea de que se pueda repartir la columna k entre el resto de procesos de la misma malla y se pueda repartir la fila k entre el resto de procesos de la misma columna, necesitamos usar comunicadores, para asignar un nuevo identificador a los procesos dentro de los comunicadores.

La misión de los comunicadores consistió en etiquetar los procesos atendiendo a las necesidades que describimos anteriormente.

Las asignaciones de etiquetas quedaron de la siguiente forma:

```c++
/* Para una matriz de 9x9 si contamos con 9 procesos para formar una malla tendríamos un reparto lógico de la siguiente forma: */
[	P0	,	P1	,	P2	]
[	P3	,	P4	,	P5	]
[	P6	,	P7	,	P8	]

/* Etiquetado vertical de los procesos, de esta manera accedemos a todos los procesos que tengan la etiqueta 0 de este comunicador, que serian P0 P3 y P6, lo mismo pasa con los de las etiquetas 1 y 2*/
[	0	,	1	,	2	]
[	0	,	1	,	2	]
[	0	,	1	,	2	]

/* Etiquetado horizontal de los procesos, de esta manera accedemos a todos los procesos que tengan la etiqueta 0 de este comunicador, que serian P0 P1 y P2, lo mismo pasa con los de las etiquetas 1 y 2*/
[	0	,	0	,	0	]
[	1	,	1	,	1	]
[	2	,	2	,	2	]
```

Para lograr que quedasen etiquetados de esta manera tuve que asignarles su nueva etiqueta con el siguiente calculo:

```c++
int idHorizontal = idProceso / sqrtP;
int idVertical = idProceso % sqrtP;
```


#### 2.1.2 Reparto de bloques
Inicialmente P0 contiene la matriz completa y procede a hacer un reparto de esta al resto de procesos asignándole a cada uno un bloque de tamaño ***N/sqrtP*** * ***N/sqrtP***. 

Como obtuvimos como resultado un **buffer de envío** durante el proceso de empaquetado, ya solo quedaba repartir estos datos que estaban preparados a cada proceso.

Para ello hice un **Scatter** :
```c++
MPI_Scatter(
	bufferSalida,                         // Valores a repartir
    sizeof(int) * tamBloque * tamBloque,  // Cantidad que se envia a cada proceso
	MPI_PACKED,                           // Tipo del dato que se enviara
	subMatriz,                            // Variable donde recibir los datos
	tamBloque * tamBloque,                // Cantidad que recibe cada proceso
     MPI_INT,                             // Tipo del dato que se recibira
     0,                                   // Proceso que reparte los datos al resto (En este caso es P0)
     MPI_COMM_WORLD
);
```
Una vez que todos los procesos tienen almacenado en su matriz local los datos se puede proceder a realizar el algoritmo.

#### 2.1.3 Indices locales a globales
Nos surge un problema a la hora de comprobar que no se itere sobre la diagonal de la **matriz ¡¡COMPLETA!!** y esto es porque estamos utilizando indices locales de cada **matriz LOCAL** de cada proceso.

Para solucionar esto necesitamos pasar esos indices locales de cada proceso a indices globales y una vez obtenidos estos podremos verificar que no estemos pasando por la diagonal de la **matriz COMPLETA**.

Podemos decir que un indice **iGlobal = i + iDesplazamiento** y que un indice **jGlobal = j + jDesplazamiento**, de forma análoga obtenemos *iLocal = i - iDesplazamiento* y *jLocal = j - jDesplazamiento*.

```c++
for(i = 0; i<tamBloque; i++)
{
	iGlobal = iLocalInicio + i;    
	for(j = 0; j<tamBloque; j++)
	{
		jGlobal = jLocalInicio + j;
		// no iterar sobre la diagonal (celdas a 0)
		if (iGlobal != jGlobal && iGlobal != k && jGlobal != k) 
		{   
			//...
		}
	}
}
```

#### 2.1.4 Acceso a los datos de las submatrices
Como comenté en el [punto 1.1.3 Acceso a los datos del vector local](#113-acceso-a-los-datos-del-vector-local) fue un engorro tener que calcular usando operaciones modulo la posición del elemento i-ésimo, j-ésimo o k-ésimo utilizando un vector contiguo, por ello en esta ocasión me decanté por usar una matriz; de esta forma los accesos a los elementos de la matriz local de cada proceso quedan de esta forma:

```c++
vikj = columnak[i] + filak[j];
vikj = min(vikj, subMatriz[i][j]);
subMatriz[i][j] = vikj;
```

#### 2.1.5 Reunir resultados
Cada proceso tiene que enviar sus datos de vuelta a la **matriz COMPLETA** que conforma la solución del problema. Para ello realicé un **Gather** pero en esta ocasión los datos de entrada es la *submatriz* y el buffer de salida es el que irá acto seguido al **Unpack**.

```c++
MPI_Gather(
	subMatriz,              // datos de entrada
	tamBloque * tamBloque,  // cantidad de datos de entrada
	MPI_INT,                // tipo de los datos de entrada
	bufferSalida,           // datos de salida
	sizeof(int) * tamBloque * tamBloque,  // cantidad de datos de salida
	MPI_PACKED,             // tipo de los datos de salida
	0,                      // proceso encargado
	MPI_COMM_WORLD          // dentro del comunicador global
);
```

Uno de los problemas que me encontré al realizar el Gather fue que desconocía que había que especificar el **sizeof(int)** en la cantidad de datos que recibe el **buffer de salida** del tipo *MPI_PACKED*, hay que indicarlo explícitamente.
Caso contrario es **submatriz** como elemento de entrada que no es necesario indicarle el *sizeof(int)* y esto es porque el tipo que tiene es *MPI_INT*.

#### 2.1.6 Desempaquetar el buffer de salida
Durante este proceso tuve que volver a definir el tipo de dato especial MPI_BLOQUE ya que **Unpack** que es la operación que se va a encargar de poner de forma ordenada los datos (que es como estaban al principio, antes de hacer el **Pack**) dentro de la matriz COMPLETA que reside en el Grafo.

Los errores que me surgieron en esta sección fueron que olvidé de nuevo el **sizeof(int)** para indicar el numero de elementos que tenía el **buffer de salida**.

Y la operación quedó así:
```c++
MPI_Unpack(
	bufferSalida,                   // datos de entrada
	sizeof(int) * nverts * nverts,  // cantidad de datos de entrada
	&posicion,                      // puntero que lleva el indice por el que va bufferSalida, cada vez que se incrementa en una cantidad del tamBloque este indice queda incrementado también
	G->getPtrMatriz() + comienzo,   // puntero a la posición donde vamos a copiar los datos ya ordenados en cada desempaquetado
	1,                              // cantidad de bloques
	MPI_BLOQUE,                     // tipo de los datos 
	MPI_COMM_WORLD
);
```



##3. Resultados
####*Información del equipo*
 - Nombre del modelo: MacBook Pro
 - Nombre del procesador: Intel Core i5
 - Velocidad del procesador: 2,7GHz
 - Cantidad de procesadores: 1
 - Cantidad total de núcleos: 2
 - Caché de nivel 2 (por núcleo): 256KB
 - Caché de nivel 3: 3MB
 - Memoria: 8GB
 - SO: Darwin Kernel Version 15.4.0 (64bits)
 - Compilador c++: Apple LLVM version 7.3.0 (clang-703.0.29)
 - Compilador MPI: mpicxx Open MPI 1.6.5 (Language: C++)

####*Tabla de tiempos y ganancia*

En la siguiente tabla se pueden observar las mediciones de tiempos que se han tomado antes y después de finalizar únicamente el algoritmo de Floyd en sus distintas versiones.

La ganancia se puede expresar como la relación entre el tiempo secuencias y el tiempo paralelo *Tsecuencial/Tparalelo*.

 - Ganancia (Floyd1D)  = FloydS/Floyd1D (P = 4)
 - Ganancia (Floyd2D)  = FloydS/Floyd2D (P = 4)

| Tamaño | P = 1 (FloydS) | P = 4 (Floyd1D) | P = 4 (Floyd2D) | Ganancia (Floyd1D) | Ganancia (Floyd2D) |
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| n = 60 | 0.000809 | 0.001683 | 0.000411 | 0.48068924539512775 | 1.9683698296836982 |
| n = 240 | 0.047861 | 0.025798 | 0.010350 | 1.8552213349872082 |  4.624251207729468 |
| n = 480 | 0.373224 | 0.179740 | 0.075400 | 2.0764660064537663 | 4.949920424403183 |
| n = 600 | 0.748859 | 0.357499 | 0.144144 | 2.09471634885692 | 5.195214507714509 |
| n = 800 | 1.794203 | 0.876496 | 0.346687 | 2.047017898541465 | 5.175282026727278 |
| n = 920 | 2.793257 | 1.326323 | 0.538065 | 2.106015653803787 | 5.191300307583656
 |
| n = 1000 | 3.571158 | 1.682442 | 0.706264 | 2.1226039292884984 | 5.056406669460712 |
| n = 1200 | 6.215353 | 3.006751 | 1.197015 | 2.067132595948251 | 5.192376870799447 |

####*Grafica de tiempos obtenidos en la tabla*
![graficaP1_1](./grafica.png)

####*Grafica de ganancia obtenidas en la tabla*
![graficaP1_2](./ganancia.png)

Como se puede observar tanto en la tabla como en las gráficas los tiempos recogidos por los algoritmos paralelos consumen mucho menos tiempo que el secuenciasl y presentan una ganancia de hasta un 420% por encima.


