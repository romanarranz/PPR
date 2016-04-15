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

all: $(BIN)/tspSecuencial $(BIN)/tspParalelo

# ************ Ejecutables ************

$(BIN)/tspSecuencial: $(OBJ)/bbseq.o $(OBJ)/libbb.o
	@echo "Creando ./bin/tspSecuencial..."
	@$(CXX) -o $@ $^

$(BIN)/tspParalelo: $(OBJ)/tspParalelo.o $(OBJ)/libbb.o
	@echo "Creando ./bin/tspParalelo..."
	@$(CXX) -o $@ $^

# ************ Objetos ************

$(OBJ)/bbseq.o: $(SRC)/bbseq.cc
	@echo "Creando ./obj/bbseq.o..."
	@$(CXX) $(CPPFLAGS) $< -o $@

$(OBJ)/tspParalelo.o: $(SRC)/tspParalelo.cpp
	@echo "Creando ./obj/tspParalelo.o..."
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
	zip $(CURRENT_DIR) -r src lib obj input include enunciados bin Makefile

tar:
	@echo "Creando "$(CURRENT_DIR)".tar.gz..."
	tar -cvzf $(CURRENT_DIR).tar.gz src lib obj input include enunciados bin Makefile