#!/bin/bash


export JULIA_CUDA_MEMORY_POOL=none
export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false
# taskset --cpu-list 0-31 mpiexecjl -np 2 julia --project=${HOME}/.julia/dev/CurvilinearDiffusion -t 4 run.jl  
mpiexecjl -np 2 \
  julia --project=${HOME}/.julia/dev/CurvilinearDiffusion -t 4 run.jl  
# mpiexecjl --project=${HOME}/.julia/dev/CurvilinearDiffusion -np 2 julia -t 6 run.jl  
