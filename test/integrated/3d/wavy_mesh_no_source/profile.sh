#!/bin/bash


jules=${HOME}/.juliaup/bin/julia
nsys profile $jules --project=${HOME}/.julia/dev/CurvilinearDiffusion -t auto run.jl