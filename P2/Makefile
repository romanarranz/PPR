###############################################################################
#
# 	 Makefile
# -----------------------------------------------------------------------------
#  	Copyright (C) 2016  	  Román Arranz Guerrero  	University of Granada
# -----------------------------------------------------------------------------
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################

SRC = src
INC = include
OBJ = obj
LIB = lib
BIN = bin
CXX = mpic++
CC  = mpicc
DEBUG = -Wall -g -O0
PROD = -O3

# cambiar $(DEBUG) por $(PROD) cuando funcione correctamente
CPPFLAGS = $(DEBUG) -I$(INC) -I$(LIB) -c

MAKEFILE_PATH = $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR = $(lastword $(notdir $(patsubst %/,%,$(dir $(MAKEFILE_PATH)))))

all: $(BIN)/tspSecuencial $(BIN)/tspParalelo $(BIN)/tspParaleloDebug

# ************ Ejecutables ************

$(BIN)/tspSecuencial: $(OBJ)/bbseq.o $(OBJ)/libbb.o
	@echo "Creando ./bin/tspSecuencial..."
	@$(CXX) -o $@ $^

$(BIN)/tspParalelo: $(OBJ)/tspParalelo.o $(OBJ)/libbb.o $(OBJ)/equilibradoCarga.o $(OBJ)/difusionCotaSuperior.o
	@echo "Creando ./bin/tspParalelo..."
	@$(CXX) -o $@ $^

$(BIN)/tspParaleloDebug: $(OBJ)/tspParalelo.o $(OBJ)/libbb.o $(OBJ)/equilibradoCarga_debug.o $(OBJ)/difusionCotaSuperior_debug.o
	@echo "Creando ./bin/tspParaleloDebug..."
	@$(CXX) -o $@ $^

# ************ Objetos ************

$(OBJ)/bbseq.o: $(SRC)/bbseq.cc
	@echo "Creando ./obj/bbseq.o..."
	@$(CXX) $(CPPFLAGS) $< -o $@

$(OBJ)/tspParalelo.o: $(SRC)/tspParalelo.cpp
	@echo "Creando ./obj/tspParalelo.o..."
	@$(CXX) $(CPPFLAGS) $< -o $@

$(OBJ)/equilibradoCarga.o: $(SRC)/equilibradoCarga.cpp
	@echo "Creando ./obj/equilibradoCarga.o..."
	@$(CXX) $(CPPFLAGS) $< -o $@

$(OBJ)/equilibradoCarga_debug.o: $(SRC)/equilibradoCarga_debug.cpp
	@echo "Creando ./obj/equilibradoCarga_debug.o..."
	@$(CXX) $(CPPFLAGS) $< -o $@

$(OBJ)/difusionCotaSuperior.o: $(SRC)/difusionCotaSuperior.cpp
	@echo "Creando ./obj/difusionCotaSuperior.o..."
	@$(CXX) $(CPPFLAGS) $< -o $@

$(OBJ)/difusionCotaSuperior_debug.o: $(SRC)/difusionCotaSuperior_debug.cpp
	@echo "Creando ./obj/difusionCotaSuperior_debug.o..."
	@$(CXX) $(CPPFLAGS) $< -o $@

$(OBJ)/libbb.o: $(LIB)/libbb.cc
	@echo "Creando ./obj/bbseq.o..."
	@$(CXX) $(CPPFLAGS) $< -o $@

.PHONY: clean zip tar plot style debug license

# ************ Limpieza ************
clean:
	@echo "Borrando ejecutables, objeto y cabeceras..."
	@rm $(BIN)/* $(OBJ)/* $(SRC)/*~ $(INC)/*~ ./*~

# ************ Compresion ************
zip:
	@echo "Creando "$(CURRENT_DIR)".zip..."
	zip $(CURRENT_DIR) -r src bin enunciados include input lib obj output src Makefile

tar:
	@echo "Creando "$(CURRENT_DIR)".tar.gz..."
	tar -cvzf $(CURRENT_DIR).tar.gz bin enunciados include input lib obj output src Makefile

# ************ Grafica ************
plot:
	@echo "Creando grafica.png..."
	@gnuplot <<< "\
		set terminal png size 800,600 font 'Verdana,10'; \
		set output 'grafica.png'; \
		set autoscale; \
		unset log; \
		unset label; \
		set border linewidth 2; \
		set xtic auto; \
		set ytic auto; \
		set title 'Viajante de Comercio'; \
		set xlabel 'Tamaño del problema'; \
		set ylabel 'Tiempo de computo (seg)'; \
		set key default; \
		set xr [0.0:35]; \
		set yr [0:4]; \
		plot 'output/tspS.dat' using 1:2 title 'TSP Secuencial' with linespoints pt 7 ps 1, \
		     'output/tspP.dat' using 1:2 title 'TSP PDistribuido' with linespoints pt 6 ps 1"
	@echo "Creando ganancia.png..."
	@gnuplot <<< "\
		set terminal png size 800,600 font 'Verdana,10'; \
		set output 'ganancia.png'; \
		set autoscale; \
		unset log; \
		unset label; \
		set border linewidth 2; \
		set xtic auto; \
		set ytic auto; \
		set title 'Ganancia TSPP Distribuido frente a Secuencial'; \
		set xlabel 'Tamaño del problema'; \
		set ylabel 'Ganancia (S)'; \
		set key default; \
		set xr [0.0:35]; \
		set yr [0:4]; \
		plot 'output/gananciaS.dat' using 1:2 title 'GananciaS' with linespoints pt 7 ps 1, \
		     'output/gananciaP.dat' using 1:2 title 'GananciaP' with linespoints pt 6 ps 1"

style:
	@echo "Haciendo una copia de los originales..."
	@if [ ! -d "src/old" ]; then mkdir src/old; fi
	@-cp src/*.cpp src/old||:
	@echo "Haciendo legible el codigo..."
	@astyle --style=allman --indent=spaces=4 --pad-header --delete-empty-lines --align-pointer=name --align-reference=middle \
	--break-closing-brackets --add-brackets --convert-tabs --close-templates --mode=c --indent-preproc-block --indent-preproc-cond \
	--indent-preproc-define --indent-col1-comments src/*.cpp

# con ||: make no nos muestra el error
# con @- make ignora los errores y no muestra la salida de la ejecucion de esta sentencia
debug:
	@echo "Ejecutando valgrind..."
	@-mpirun -np 4 valgrind --tool=memcheck --leak-check=full -v bin/tspParalelo 10 input/tsp10.1 &> output/debug_full.txt ||:
	@echo "Filtrando resultados por .cpp..."
	@grep ".cpp" output/debug_full.txt > output/debug.txt
	@echo "Ordenando y borrando lineas repetidas..."
	@sort -u output/debug.txt -o output/debug.txt
	@echo "Salida: output/debug.txt, output/debug_full.txt"

license:
	@echo "Generando licencia..."
	@license install mit -y 2016 -n "ROMÁN ARRANZ GUERRERO"
