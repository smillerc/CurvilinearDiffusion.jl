module ADESolvers

using .Threads
using BlockHaloArrays
using CurvilinearGrids
using Polyester, StaticArrays
using UnPack
using ..Partitioning

export ADESolver
export BlockADESolver
export solve!, update_conductivity!, update_mesh_metrics!

struct ADESolver{T,N,EM,F,NT,BC}
  qⁿ⁺¹::Array{T,N}
  pⁿ⁺¹::Array{T,N}
  J::Array{T,N} # cell-centered Jacobian
  metrics::Array{EM,N}
  a₋ⁿ⁺¹::Array{T,N} # cell-centered diffusivity
  a₊ⁿ⁺¹::Array{T,N} # cell-centered diffusivity
  aⁿ⁺¹::Array{T,N} # cell-centered diffusivity
  source_term::Array{T,N} # cell-centered source term
  mean_func::F
  limits::NT
  bcs::BC
  nhalo::Int
  conservative::Bool # uses the conservative form
end

include("averaging.jl")
include("boundary_conditions.jl")
include("mesh_metrics.jl")
include("conductivity.jl")
include("ADESolver.jl")
include("BlockADESolver.jl")

end
