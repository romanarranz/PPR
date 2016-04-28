#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# Delay entre imagen e imagen de la animcacion
delay = raw_input("Introduce el retardo entre imagenes: ");
delay = int(delay)
if delay <= 0:
    print "El retardo introducido debe ser mayor a 0"
else:
    # Seleccionar la secuencia de imagnes que se quiere animar
    print "Listado de imagenes .gif disponibles para animar en este directorio"
    listaImagenes = []
    for file in os.listdir("."):
        if file.endswith(".gif"):
            listaImagenes.append(file)
            print "\t" + str(len(listaImagenes)-1) + "- " + file

    print "Indica el orden de imagenes que quieras para crear la animacion"
    print "Por ejemplo -> 0,2,3"
    ordenAnimacion = raw_input("Orden: ")
    ordenAnimacion = ordenAnimacion.split(',')

    # Recorro todos los indices seleccionados y los añado a otra lista
    archivosEntrada = ""
    for i in ordenAnimacion:
        archivosEntrada = archivosEntrada + listaImagenes[int(i)] + " "

    archivoSalida = raw_input("Nombre archivo de salida: ")

    # Realizamos la animacion con los parametros indicados
    os.system("gifsicle --delay=" + str(delay) + " --loop " + archivosEntrada + " > "+archivoSalida)
