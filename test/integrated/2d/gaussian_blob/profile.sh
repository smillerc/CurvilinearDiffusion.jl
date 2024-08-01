#!/bin/bash

export OPENBLAS_NUM_THREADS=1
export JULIA_CUDA_MEMORY_POOL=none
export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false
# taskset --cpu-list 0-31 mpiexecjl -np 2 julia --project=${HOME}/.julia/dev/CurvilinearDiffusion -t 4 run.jl  


nsys profile \
 --stats=true --trace=nvtx,mpi,cuda \
 --mpi-impl=mpich --force-overwrite=true \
 /opt/mpich/4.2.2/bin/mpirun -np 2 \
  julia --project=${HOME}/.julia/dev/CurvilinearDiffusion -t 4 run.jl > nsys_profile.output 2>&1
