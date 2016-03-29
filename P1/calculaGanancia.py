#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import argv
import re

# Si el numero de parametros del programa es menor que 3 o los parametros primero y segundo son el mismo archivo
if len(argv) < 3 or argv[1] == argv[2]:
	print "Error la sintaxis es:"
	print "\t$",argv[0]," output/floydS.dat"," output/floyd1D.dat"
	print "\t$",argv[0]," output/floydS.dat"," output/floyd2D.dat"
else:
	archivoFloydS = argv[1]
	archivoFloydP = argv[2]

	flujoArchivoS = open(archivoFloydS)
	flujoArchivoP = open(archivoFloydP)

	# creo un diccionario vacio
	ganancia = {}

	# <== Me quedo con los tiempos del archivo secuencial
	# =========================================>
	print "Flujo de %r:" % archivoFloydS
	# Para cada linea
	for linea in flujoArchivoS:
		# me creo una lista usando como delimitador el caracter '\t'
		arrayLinea = re.split(r'\t+', linea.rstrip('\t'))

		# reemplazo en cada elemento de la lista el salto de linea por la cadena vacia
		arrayLinea = ([elemento.replace('\n', '') for elemento in arrayLinea])
		if arrayLinea:
			print "\tarrayLinea: ", arrayLinea
			clave = int(arrayLinea[0])
			ganancia[clave] = float(arrayLinea[1])
		else:
			print "\tNo match"
	flujoArchivoS.close()

	print ""

	# <== Me quedo con los tiempos del archivo paralelo
	# =========================================>
	print "Flujo de %r:" % archivoFloydP
	# Para cada linea
	for linea in flujoArchivoP:
		# me creo una lista usando como delimitador el caracter '\t'
		arrayLinea = re.split(r'\t+', linea.rstrip('\t'))

		# reemplazo en cada elemento de la lista el salto de linea por la cadena vacia
		arrayLinea = ([elemento.replace('\n', '') for elemento in arrayLinea])
		if arrayLinea:
			print "\tarrayLinea: ", arrayLinea
			clave = int(arrayLinea[0])
			# divido el tiempo secuencial entre el tiempo paralelo y lo guardo como valor del diccionario
			ganancia[clave] = ganancia[clave]/float(arrayLinea[1])
		else:
			print "\tNo match"
	flujoArchivoP.close()

	print ""

	# <== Imprimo el diccionario
	# =========================================>
	print "Diccionario ganancia"
	for key, value in sorted(ganancia.iteritems()):
		print "\t",key, value

	# <== Guardo el diccionario en un fichero ganancia.dat
	# =========================================>
	archivoSalida = "ganancia"+archivoFloydP[-6:]
	flujoSalida = open("output/"+archivoSalida, 'w')

	for key, value in sorted(ganancia.iteritems()):
		linea = str(key)+'\t'+str(value)+'\n'
		s = str(linea)
		flujoSalida.write(s)
	flujoSalida.close()