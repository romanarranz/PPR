#!/bin/bash

export OMP_NUM_THREADS=4
mpirun -np 2 -x OMP_NUM_THREADS bin/hello
