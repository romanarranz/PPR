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
# MPI
MPICFLAGS = $(shell mpicc -showme:compile)
MPILDFLAGS = $(shell mpicc -showme:link) -lmpi -lm

# OpenMP
OPENMP = -fopenmp

SRC = src
INC = include
OBJ = obj
BIN = bin
CXX = g++-5
CC  = gcc-5
CFLAGS = -Wall -g -I$(INC) $(MPICFLAGS) $(OPENMP) -O3
LDFLAGS= -L/usr/local/lib $(MPILDFLAGS)

MAKEFILE_PATH = $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR = $(lastword $(notdir $(patsubst %/,%,$(dir $(MAKEFILE_PATH)))))

all: $(BIN)/hello

# ************ Ejecutables ************

$(BIN)/hello: $(OBJ)/hello.o
	@echo "Creando ./bin/hello..."
	@$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

# ************ Objetos ************

$(OBJ)/hello.o: $(SRC)/hello.c
	@echo "Creando ./obj/hello.o..."
	@$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean zip tar

# ************ Limpieza ************
clean:
	@echo "Borrando ejecutables, objeto y cabeceras..."
	@rm $(BIN)/* $(OBJ)/* $(SRC)/*~ $(INC)/*~ ./*~

# ************ Compresion ************
zip:
	@echo "Creando "$(CURRENT_DIR)".zip..."
	zip $(CURRENT_DIR) -r src obj include bin

tar:
	@echo "Creando "$(CURRENT_DIR)".tar.gz..."
	tar -cvzf $(CURRENT_DIR).tar.gz src obj include bin
