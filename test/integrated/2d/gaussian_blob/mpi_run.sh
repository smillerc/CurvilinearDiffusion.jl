#!/bin/bash

# taskset --cpu-list 0-31 mpiexecjl -np 2 julia --project=${HOME}/.julia/dev/CurvilinearDiffusion -t 4 run.jl  
 numactl --physcpubind=0-7 mpiexecjl -np 2 julia --project=${HOME}/.julia/dev/CurvilinearDiffusion -t 4 run.jl  
# mpiexecjl --project=${HOME}/.julia/dev/CurvilinearDiffusion -np 2 julia -t 6 run.jl  
