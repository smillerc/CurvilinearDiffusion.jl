#!/bin/bash

export OPENBLAS_NUM_THREADS=1


# mpirun -np 2  julia --project=${HOME}/.julia/dev/CurvilinearDiffusion -t 6 run.jl

# nsys profile --stats=true --trace=nvtx,mpi --mpi-impl=mpich --force-overwrite=true mpiexecjl --project -np 6 julia -t 1 mpi_trapezoidal.jl  > nsys_profile.output 2>&1
nsys profile \
 --stats=true --trace=nvtx,mpi,cuda \
 --mpi-impl=openmpi --force-overwrite=true \
 mpiexecjl -np 2 julia \
 --project=${HOME}/.julia/dev/CurvilinearDiffusion -t 1 run.jl  > nsys_profile.output 2>&1
# nsys profile --stats=true --trace=nvtx,mpi --mpi-impl=openmpi --force-overwrite=true mpiexecjl --project=${HOME}/.julia/dev/CurvilinearDiffusion julia -t 6 run.jl  > nsys_profile.output 2>&1
