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

# ************ Deprecated Interface ************
ifeq ($(x86_64),1)
	@echo "x86_64 variable has been deprecated"
    @echo "please use TARGET_ARCH=x86_64 instead"
    TARGET_ARCH ?= x86_64
endif

# ************ Architecture ************
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
    TARGET_SIZE := 64
else ifeq ($(TARGET_ARCH),armv7l)
    TARGET_SIZE := 32
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif
ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-ppc64le))
        $(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
    endif
endif

# ************ OS ************
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin qnx android))
    $(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

# OpenMP
OPENMP := -fopenmp

SRC := src
INC :=
OBJ := obj
BIN := bin
LIB :=
CXX := g++-5
CC  := gcc-5

DEBUG := -Wall -g
PROD := -O3

CFLAGS := $(PROD)
LDFLAGS := -L/usr/local/lib

MAKEFILE_PATH = $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR = $(lastword $(notdir $(patsubst %/,%,$(dir $(MAKEFILE_PATH)))))

all: binaries cleanup
binaries: main

# ************ Ejecutables ************

main: $(OBJ)/main.o
	@echo "Creando $@..."
	@mkdir -p $(BIN)/$(TARGET_ARCH)/$(TARGET_OS)
	@$(CXX) $(CFLAGS) $(OPENMP) $(LDFLAGS) -o $@ $+
	@mv $@ $(BIN)/$(TARGET_ARCH)/$(TARGET_OS)

# ************ Objetos ************
$(OBJ)/main.o: $(SRC)/main.cpp
	@echo "Creando $@..."
	@$(CXX) $(CFLAGS) $(OPENMP) -c $< -o $@

# ************ Clean UP ************
cleanup:
ifeq ($(TARGET_OS),darwin)
	@rm -r *.dSYM
endif

.PHONY: clean zip tar plot

# ************ Limpieza ************
clean:
	@echo "Borrando ejecutables, objeto y cabeceras..."
	@rm -r $(BIN)/* $(OBJ)/* $(SRC)/*~ $(INC)/*~ ./*~

# ************ Compresion ************
zip:
	@echo "Creando "$(CURRENT_DIR)".zip..."
	zip $(CURRENT_DIR) -r src bin obj output src Makefile

tar:
	@echo "Creando "$(CURRENT_DIR)".tar.gz..."
	tar -cvzf $(CURRENT_DIR).tar.gz src bin obj output src Makefile

# ************ Grafica ************
plot: SHELL:=/bin/bash
plot:
	@echo "Creando floydOpenMP.gif..."
	@gnuplot <<< "\
		set terminal gif size 800,600 font 'Verdana,10'; \
		set output 'floydOpenMP.gif'; \
		set autoscale; \
		unset log; \
		unset label; \
		set border linewidth 2; \
		set xtic auto; \
		set ytic auto; \
		set title 'Floyd Seq vs OpenMP'; \
		set xlabel 'Tamaño del problema'; \
		set ylabel 'Tiempo de computo (seg)'; \
		set key default; \
		set xr [0.0:1024]; \
		set yr [0:6]; \
		plot 'output/floydS.dat' using 1:2 title 'Floyd S' with linespoints pt 7 ps 1, \
			 'output/floyd1D.dat' using 1:2 title 'Floyd1D OpenMP' with linespoints pt 6 ps 1, \
			 'output/floyd2D.dat' using 1:2 title 'Floyd2D OpenMP' with linespoints pt 5 ps 1, \
			 'output/floyd1D_ilex.dat' using 1:2 title 'Floyd1D Ilex OpenMP' with linespoints pt 4 ps 1, \
		     'output/floyd2D_ilex.dat' using 1:2 title 'Floyd2D Ilex OpenMP' with linespoints pt 3 ps 1"
	@echo "Creando gananciaFloyd.gif..."
	@gnuplot <<< "\
		set terminal gif size 800,600 font 'Verdana,10'; \
		set output 'gananciaFloyd.gif'; \
		set autoscale; \
		unset log; \
		unset label; \
		set border linewidth 2; \
		set xtic auto; \
		set ytic auto; \
		set title 'Ganancia Floyd OpenMP'; \
		set xlabel 'Tamaño del problema'; \
		set ylabel 'Ganancia (S)'; \
		set key default; \
		set xr [0.0:1024]; \
		set yr [0:22]; \
		plot 'output/gananciaS.dat' using 1:2 title 'GananciaS' with linespoints pt 7 ps 1, \
			 'output/gananciaS_1D.dat' using 1:2 title 'Ganancia1D OpenMP' with linespoints pt 6 ps 1, \
			 'output/gananciaS_2D.dat' using 1:2 title 'Ganancia2D OpenMP' with linespoints pt 5 ps 1, \
			 'output/gananciaS_1D_ilex.dat' using 1:2 title 'Ganancia1D Ilex OpenMP' with linespoints pt 4 ps 1, \
			 'output/gananciaS_2D_ilex.dat' using 1:2 title 'Ganancia2D Ilex OpenMP' with linespoints pt 3 ps 1"
