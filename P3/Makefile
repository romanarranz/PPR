# CUDA Toolkit
CUDA_PATH ?= "/usr/local/cuda-7.5"
CUDA_INCLUDE := /usr/local/cuda/include
SRC := src
CU := cuda
INC := include
OBJ := obj
LIB := -lm
BIN := bin
CXX := g++
CC  := gcc

# ************ Deprecated Interface ************
ifeq ($(x86_64),1)
    $(info WARNING - x86_64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=x86_64 instead)
    TARGET_ARCH ?= x86_64
endif
ifeq ($(ARMv7),1)
    $(info WARNING - ARMv7 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=armv7l instead)
    TARGET_ARCH ?= armv7l
endif
ifeq ($(aarch64),1)
    $(info WARNING - aarch64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=aarch64 instead)
    TARGET_ARCH ?= aarch64
endif
ifeq ($(ppc64le),1)
    $(info WARNING - ppc64le variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=ppc64le instead)
    TARGET_ARCH ?= ppc64le
endif
ifneq ($(GCC),)
    $(info WARNING - GCC variable has been deprecated)
    $(info WARNING - please use CXX=$(GCC) instead)
    CXX ?= $(GCC)
endif
ifneq ($(abi),)
    $(error ERROR - abi variable has been removed)
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

# ************ Host Compiler ************
ifeq ($(TARGET_OS),darwin)
    ifeq ($(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5),1)
        CXX ?= clang++
    endif
else ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
        ifeq ($(TARGET_OS),linux)
            CXX ?= arm-linux-gnueabihf-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            CXX ?= $(QNX_HOST)/usr/bin/arm-unknown-nto-qnx6.6.0eabi-g++
        else ifeq ($(TARGET_OS),android)
            CXX ?= arm-linux-androideabi-g++
        endif
    else ifeq ($(TARGET_ARCH),aarch64)
        ifeq ($(TARGET_OS), linux)
            CXX ?= aarch64-linux-gnu-g++
        else ifeq ($(TARGET_OS), android)
            CXX ?= aarch64-linux-android-g++
        endif
    else ifeq ($(TARGET_ARCH),ppc64le)
        CXX ?= powerpc64le-linux-gnu-g++
    endif
endif
CXX ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

# ************ Flags ************
DEBUG := -Wall -g -O0
PROD := -O3
# cambiar $(DEBUG) por $(PROD) cuando funcione correctamente
CCFLAGS := $(PROD) -I$(INC) -I$(LIB) -c
LDFLAGS	:=

NVCCFLAGS   := -m${TARGET_SIZE} -D_FORCE_INLINES

MAKEFILE_PATH = $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR = $(lastword $(notdir $(patsubst %/,%,$(dir $(MAKEFILE_PATH)))))

# build flags
ifeq ($(TARGET_OS),darwin)
    LDFLAGS += -rpath $(CUDA_PATH)/lib
    CCFLAGS += -arch $(HOST_ARCH)
else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
    LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
    CCFLAGS += -mfloat-abi=hard
else ifeq ($(TARGET_OS),android)
    LDFLAGS += -pie
    CCFLAGS += -fpie -fpic -fexceptions
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
        ifneq ($(TARGET_FS),)
            GCCVERSIONLTEQ46 := $(shell expr `$(CXX) -dumpversion` \<= 4.6)
            ifeq ($(GCCVERSIONLTEQ46),1)
                CCFLAGS += --sysroot=$(TARGET_FS)
            endif
            LDFLAGS += --sysroot=$(TARGET_FS)
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
        endif
    endif
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Gencode arguments
SMS ?= 20 30 35 37 50 52

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

LIBSIZE :=
ifneq ($(TARGET_OS),darwin)
ifeq ($(TARGET_SIZE),64)
LIBSIZE := 64
endif
endif

LIBRARIES += -L$(CUDA_PATH)/lib$(LIBSIZE) -lcudart

#####################################################################################################

all: floydSecuencial floydCuda1D floydCuda1DSharedM floydCuda2D floydCuda2DSharedM creaEjemplo

# ************ Ejecutables ************

creaEjemplo: $(OBJ)/creaEjemplo.o
	@echo "Creando $@..."
	@$(CXX) -o $@ $+ $(LIBRARIES)
	@mkdir -p $(BIN)/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
	@mv $@ $(BIN)/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)

floydCuda1D: $(OBJ)/floydParalelo1D.o $(OBJ)/floydCUDA.o $(OBJ)/Graph.o
	@echo "Creando $@..."
	@$(CXX) -o $@ $+ $(LIBRARIES)
	@mkdir -p $(BIN)/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
	@mv $@ $(BIN)/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)

floydCuda1DSharedM: $(OBJ)/floydParalelo1DSharedM.o $(OBJ)/floydSharedMCUDA.o $(OBJ)/Graph.o
	@echo "Creando $@..."
	@$(CXX) -o $@ $+ $(LIBRARIES)
	@mkdir -p $(BIN)/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
	@mv $@ $(BIN)/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)

floydCuda2D: $(OBJ)/floydParalelo2D.o $(OBJ)/floydCUDA.o $(OBJ)/Graph.o
	@echo "Creando $@..."
	@$(CXX) -o $@ $+ $(LIBRARIES)
	@mkdir -p $(BIN)/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
	@mv $@ $(BIN)/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)

floydCuda2DSharedM: $(OBJ)/floydParalelo2DSharedM.o $(OBJ)/floydSharedMCUDA.o $(OBJ)/Graph.o
	@echo "Creando $@..."
	@$(CXX) -o $@ $+ $(LIBRARIES)
	@mkdir -p $(BIN)/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
	@mv $@ $(BIN)/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)

floydSecuencial: $(OBJ)/floydSecuencial.o $(OBJ)/Graph.o
	@echo "Creando $@..."
	@$(CXX) -o $@ $+ $(LIBRARIES)
	@mkdir -p $(BIN)/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
	@mv $@ $(BIN)/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)

# ************ Objetos ************

$(OBJ)/creaEjemplo.o: $(SRC)/creaEjemplo.cpp
	@echo "Creando $@..."
	@$(CXX) $(CCFLAGS) $< -o $@

$(OBJ)/Graph.o: $(SRC)/Graph.cpp
	@echo "Creando $@..."
	@$(CXX) $(CCFLAGS) $< -o $@

$(OBJ)/floydSecuencial.o: $(SRC)/floydSecuencial.cpp
	@echo "Creando $@..."
	@$(CXX) $(CCFLAGS) $< -o $@

$(OBJ)/floydParalelo1DSharedM.o: $(SRC)/floydParalelo1DSharedM.cpp
	@echo "Creando $@..."
	@$(NVCC) -I$(INC) -I$(CUDA_INCLUDE) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(OBJ)/floydParalelo1D.o: $(SRC)/floydParalelo1D.cpp
	@echo "Creando $@..."
	@$(NVCC) -I$(INC) -I$(CUDA_INCLUDE) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(OBJ)/floydParalelo2D.o: $(SRC)/floydParalelo2D.cpp
	@echo "Creando $@..."
	@$(NVCC) -I$(INC) -I$(CUDA_INCLUDE) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(OBJ)/floydParalelo2DSharedM.o: $(SRC)/floydParalelo2DSharedM.cpp
	@echo "Creando $@..."
	@$(NVCC) -I$(INC) -I$(CUDA_INCLUDE) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(OBJ)/floydCUDA.o: $(CU)/floydCUDA.cu
	@echo "Creando $@..."
	@$(NVCC) -I$(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(OBJ)/floydSharedMCUDA.o: $(CU)/floydSharedMCUDA.cu
	@echo "Creando $@..."
	@$(NVCC) -I$(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

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
plot: SHELL:=/bin/bash
plot:
	@echo "Creando floydCUDA.gif..."
	@gnuplot <<< "\
		set terminal gif size 800,600 font 'Verdana,10'; \
		set output 'floydCUDA.gif'; \
		set autoscale; \
		unset log; \
		unset label; \
		set border linewidth 2; \
		set xtic auto; \
		set ytic auto; \
		set title 'Floyd Seq vs CUDA'; \
		set xlabel 'Tamaño del problema'; \
		set ylabel 'Tiempo de computo (seg)'; \
		set key default; \
		set xr [0.0:1024]; \
		set yr [0:6]; \
		plot 'output/floydS.dat' using 1:2 title 'Floyd S' with linespoints pt 7 ps 1, \
		     'output/floyd1D.dat' using 1:2 title 'Floyd1D CUDA' with linespoints pt 6 ps 1, \
			 'output/floyd1DShared.dat' using 1:2 title 'Floyd1DShared CUDA' with linespoints pt 5 ps 1, \
			 'output/floyd2D.dat' using 1:2 title 'Floyd1D CUDA' with linespoints pt 4 ps 1, \
			 'output/floyd2DShared.dat' using 1:2 title 'Floyd2DShared CUDA' with linespoints pt 3 ps 1"
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
		set title 'Ganancia TSP con Difusion de Cota'; \
		set xlabel 'Tamaño del problema'; \
		set ylabel 'Ganancia (S)'; \
		set key default; \
		set xr [0.0:1024]; \
		set yr [0:75]; \
		plot 'output/gananciaS.dat' using 1:2 title 'GananciaS' with linespoints pt 7 ps 1, \
		     'output/gananciaS_1D.dat' using 1:2 title 'Ganancia1D' with linespoints pt 6 ps 1, \
			 'output/gananciaS_1DShared.dat' using 1:2 title 'Ganancia1DShared' with linespoints pt 5 ps 1, \
			 'output/gananciaS_2D.dat' using 1:2 title 'Ganancia1D' with linespoints pt 4 ps 1, \
			 'output/gananciaS_2DShared.dat' using 1:2 title 'Ganancia2DShared' with linespoints pt 3 ps 1,"

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
