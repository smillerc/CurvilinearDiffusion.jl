module ADESolvers

using .Threads
using BlockHaloArrays
using CurvilinearGrids
using Polyester, StaticArrays
using KernelAbstractions
using UnPack
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

function check_diffusivity_validity(scheme)
  @kernel function _kernel(α, corners)
    idx = @index(Global, Cartesian)

    @inbounds begin
      if !isfinite(α[idx]) || α[idx] < 0
        if !in(idx, corners)
          error("Invalid diffusivity α=$(α[idx]) at $idx")
        end
      end
    end
  end
end

function validate_diffusivity(solver::AbstractADESolver{N,T}) where {N,T}
  nhalo = 1

  domain = solver.iterators.domain.cartesian

  α_domain = @view solver.α[domain]

  domain_valid = (all(isfinite.(α_domain)) && all(map(x -> x >= 0, α_domain)))

  if !domain_valid
    error("Invalid diffusivity in the domain")
  end

  for axis in 1:N
    bc = haloedge_regions(domain, axis, nhalo)
    lo_edge = bc.halo.lo
    hi_edge = bc.halo.hi

    α_lo = @view solver.α[lo_edge]
    α_hi = @view solver.α[hi_edge]

    α_lo_valid = (all(isfinite.(α_lo)) && all(map(x -> x >= 0, α_lo)))
    α_hi_valid = (all(isfinite.(α_hi)) && all(map(x -> x >= 0, α_hi)))

    if !α_lo_valid
      error("Invalid diffusivity in the lo halo region for axis: $axis")
    end

    if !α_hi_valid
      error("Invalid diffusivity in the hi halo region for axis: $axis")
    end
  end
end

end
