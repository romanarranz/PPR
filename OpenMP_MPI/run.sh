#!/bin/bash

mpirun -np 2 -x OMP_NUM_THREADS bin/hello
