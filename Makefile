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
BIN = bin
CXX = mpic++
CC  = mpicc
CPPFLAGS = -Wall -g -I$(INC) -O3 -c

all: $(BIN)/floyd $(BIN)/crearEjemplos

# ************ Ejecutables ************
$(BIN)/floyd: $(OBJ)/Graph.o $(OBJ)/floyd.o
	@echo "Creando ./bin/floyd..."
	@$(CXX) -o $@ $^

$(BIN)/crearEjemplos: $(OBJ)/Graph.o $(OBJ)/crearEjemplos.o
	@echo "Creando ./bin/crearEjemplos..."
	@$(CXX) -o $@ $^

# ************ Objetos ************

$(OBJ)/Graph.o: $(SRC)/Graph.cpp
	@echo "Creando ./obj/Graph.o..."
	@$(CXX) $(CPPFLAGS) $< -o $@

$(OBJ)/floyd.o: $(SRC)/floyd.cpp
	@echo "Creando ./obj/floyd.o..."
	@$(CXX) $(CPPFLAGS) $< -o $@

$(OBJ)/crearEjemplos.o: $(SRC)/crearEjemplos.cpp
	@echo "Creando ./obj/crearEjemplos.o..."
	@$(CXX) $(CPPFLAGS) $< -o $@

.PHONY: clean

# ************ Limpieza ************
clean:
	@echo "Borrando ejecutables, objeto y cabeceras..."
	@rm $(BIN)/* $(OBJ)/* $(SRC)/*~ $(INC)/*~ ./*~
