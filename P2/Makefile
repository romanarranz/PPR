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
CPPFLAGS = -Wall -g -I$(INC) -I$(LIB) -O3 -c

MAKEFILE_PATH = $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR = $(lastword $(notdir $(patsubst %/,%,$(dir $(MAKEFILE_PATH)))))

all: $(BIN)/tspSecuencial

# ************ Ejecutables ************

$(BIN)/tspSecuencial: $(OBJ)/bbseq.o $(OBJ)/libbb.o
	@echo "Creando ./bin/tspSecuencial..."
	@$(CXX) -o $@ $^

# ************ Objetos ************

$(OBJ)/bbseq.o: $(SRC)/bbseq.cc
	@echo "Creando ./obj/bbseq.o..."
	@$(CXX) $(CPPFLAGS) $< -o $@

$(OBJ)/libbb.o: $(LIB)/libbb.cc
	@echo "Creando ./obj/bbseq.o..."
	@$(CXX) $(CPPFLAGS) $< -o $@

.PHONY: clean zip tar plot

# ************ Limpieza ************
clean:
	@echo "Borrando ejecutables, objeto y cabeceras..."
	@rm $(BIN)/* $(OBJ)/* $(SRC)/*~ $(INC)/*~ ./*~

# ************ Compresion ************
zip:
	@echo "Creando "$(CURRENT_DIR)".zip..."
	zip $(CURRENT_DIR) -r src obj input include enunciados bin

tar:
	@echo "Creando "$(CURRENT_DIR)".tar.gz..."
	tar -cvzf $(CURRENT_DIR).tar.gz src obj input include enunciados bin

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
		set title 'Algoritmo de Floyd'; \
		set xlabel 'Tamaño del problema'; \
		set ylabel 'Tiempo de computo (seg)'; \
		set key default; \
		set xr [0.0:1200]; \
		set yr [0:7]; \
		plot 'output/floydS.dat' using 1:2 title 'FloydS' with linespoints pt 7 ps 1, \
		     'output/Floyd1D.dat' using 1:2 title 'Floyd1D' with linespoints pt 6 ps 1, \
		     'output/Floyd2D.dat' using 1:2 title 'Floyd2D' with linespoints pt 7 ps 1"

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
		set title 'Ganancia Floyd Paralelo frente a Secuencial'; \
		set xlabel 'Tamaño del problema'; \
		set ylabel 'Ganancia (S)'; \
		set key default; \
		set xr [0.0:1200]; \
		set yr [0:6]; \
		plot 'output/ganancia1D.dat' using 1:2 title 'Ganancia1D' with linespoints pt 7 ps 1, \
		     'output/ganancia2D.dat' using 1:2 title 'Ganancia2D' with linespoints pt 6 ps 1"