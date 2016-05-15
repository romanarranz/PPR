# CUDA code generation flags
ifneq ($(OS_ARCH),armv7l)
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
endif
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := #-gencode arch=compute_50,code=sm_50
GENCODE_SMXX    := #-gencode arch=compute_50,code=compute_50
GENCODE_FLAGS   ?= $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM32) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SMXX)


floyd_template: floyd_template.cu Graph.cc Graph.h
	nvcc  -O3 -m64   $(GENCODE_FLAGS)   floyd_template.cu Graph.cc -o floyd

clean:
	rm -f floyd floyd2 floyd2d *.o. *.bin

