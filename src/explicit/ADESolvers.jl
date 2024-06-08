module ADESolvers

using .Threads
using BlockHaloArrays
using CurvilinearGrids
using Polyester, StaticArrays
using KernelAbstractions
using UnPack
using TimerOutputs
using Printf
# using ..Partitioning

export AbstractADESolver, ADESolver, ADESolverNSweep, BlockADESolver
export solve!, validate_diffusivity

abstract type AbstractADESolver{N,T} end

struct ADESolver{N,T,AA<:AbstractArray{T,N},F,BC,IT,L,BE} <: AbstractADESolver{N,T}
  uⁿ⁺¹::AA
  qⁿ⁺¹::AA
  pⁿ⁺¹::AA
  α::AA # cell-centered diffusivity
  source_term::AA # cell-centered source term
  mean_func::F
  bcs::BC
  iterators::IT
  limits::L
  backend::BE # GPU / CPU
  nhalo::Int
end

struct ADESolverNSweep{N,T,N2,AA<:AbstractArray{T,N},F,BC,IT,L,BE} <: AbstractADESolver{N,T}
  uⁿ⁺¹::AA
  usweepᵏ::NTuple{N2,AA}
  α::AA # cell-centered diffusivity
  source_term::AA # cell-centered source term
  mean_func::F
  bcs::BC
  iterators::IT
  limits::L
  backend::BE # GPU / CPU
end

include("../averaging.jl")
include("../edge_terms.jl")
include("boundary_conditions.jl")
include("ADESolver.jl")
include("ADESolverNSweep.jl")

function cutoff!(a)
  backend = KernelAbstractions.get_backend(a)
  cutoff_kernel!(backend)(a; ndrange=size(a))
  return nothing
end

@kernel function cutoff_kernel!(a)
  idx = @index(Global, Linear)

  @inbounds begin
    _a = cutoff(a[idx])
    a[idx] = _a
  end
end

end
