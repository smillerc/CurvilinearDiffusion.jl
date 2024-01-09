module ImplicitSchemeType

using LinearSolve
using CurvilinearGrids
using Polyester, StaticArrays
using UnPack
using TimerOutputs
using SparseArrays
using StaticArrays
using AlgebraicMultigrid
using Krylov
# using KrylovKit
using LinearSolve
using ILUZero
using IncompleteLU
using LinearAlgebra
using OffsetArrays

export ImplicitScheme, solve!, assemble_matrix!

struct ImplicitScheme{N,LP,SM,V,ST,T,EM,F,BC,CI1,CI2}
  prob::LP # linear problem (contains A, b, u0, solver...)
  A::SM # sparse matrix
  x::V # solution vector
  u0::V
  b::V # RHS vector
  solver::ST # linear solver, e.g. GMRES, CG, etc.
  J::Array{T,N} # cell-centered Jacobian
  metrics::Array{EM,N}
  α::Array{T,N} # cell-centered diffusivity
  source_term::Array{T,N} # cell-centered source term
  mean_func::F
  bcs::BC
  nhalo::Int
  conservative::Bool # uses the conservative form
  domain_indices::CI1
  halo_aware_indices::CI2
end

include("mesh_metrics.jl")
include("averaging.jl")
include("matrix_assembly.jl")

function ImplicitScheme(
  mesh::CurvilinearGrid2D,
  bcs;
  form=:conservative,
  mean_func=arithmetic_mean,
  T=Float64,
  solver=KrylovJL_GMRES(),
)
  celldims = cellsize_withhalo(mesh)

  ni, nj = cellsize(mesh)
  len = ni * nj

  fullCI = CartesianIndices(cellsize_withhalo(mesh))
  halo_aware_CI = fullCI[
    (begin + mesh.nhalo):(end - mesh.nhalo), (begin + mesh.nhalo):(end - mesh.nhalo)
  ]
  domain_CI = CartesianIndices(size(halo_aware_CI))

  @assert length(domain_CI) == length(halo_aware_CI)

  #! format: off
  A = spdiagm(
    -ni - 1 => zeros(len - ni - 1), # (i-1, j-1)
    -ni     => zeros(len - ni),     # (i  , j-1)
    -ni + 1 => zeros(len - ni + 1), # (i+1, j-1)
    -1      => zeros(len - 1),      # (i-1, j  )
    0       => ones(len),          # (i  , j  )
    1       => zeros(len - 1),      # (i+1, j  )
    ni - 1  => zeros(len - ni + 1), # (i-1, j+1)
    ni      => zeros(len - ni),     # (i  , j+1)
    ni + 1  => zeros(len - ni - 1), # (i+1, j+1)
  )
  #! format: on

  if form === :conservative
    conservative = true
    metric_type = typeof(_conservative_metrics(mesh, (1, 1)))
  else
    conservative = false
    metric_type = typeof(_non_conservative_metrics(mesh, (1, 1)))
  end

  b = zeros(T, len)
  u0 = zeros(T, len) # initial guess for iterative solvers
  x = zeros(T, len)
  edge_metrics = Array{metric_type,2}(undef, celldims)

  diffusivity = zeros(T, celldims)
  source_term = zeros(T, celldims)
  J = zeros(T, celldims)

  # prob = LinearProblem(A, b)

  # ml = ruge_stuben(prob.A) # Construct a Ruge-Stuben solver
  # pl = aspreconditioner(ml)
  # alg = KrylovKitJL_GMRES
  # alg = KrylovJL(;
  #   KrylovAlg=Krylov.gmres!, Pl=nothing, Pr=nothing, gmres_restart=0, window=0
  # )

  # linear_problem = init(prob, alg; alias_A=true, alias_b=true)
  # linear_problem = init(prob, KrylovJL_GMRES(); alias_A=false, alias_b=false)

  linear_problem = nothing # LinearProblem(A, b; u0=u0)

  implicit_solver = ImplicitScheme(
    linear_problem,
    A,
    x,
    u0,
    b,
    solver,
    J,
    edge_metrics,
    diffusivity,
    source_term,
    mean_func,
    bcs,
    mesh.nhalo,
    conservative,
    domain_CI,
    halo_aware_CI,
  )
  update_mesh_metrics!(implicit_solver, mesh)

  return implicit_solver
end

function solve!(scheme::ImplicitScheme, ::CurvilinearGrids.AbstractCurvilinearGrid, u, Δt)
  domain_LI = LinearIndices(scheme.domain_indices)

  # update the A matrix
  @timeit "assemble_matrix" assemble_matrix!(scheme, u, Δt)

  # @timeit "solve_step" LinearSolve.solve!(scheme.prob)

  # for (grid_idx, mat_idx) in zip(scheme.halo_aware_indices, domain_LI)
  #   u[grid_idx] = scheme.prob.u[mat_idx]
  # end

  # use the current solution as a guess for the next
  for (grid_idx, mat_idx) in zip(scheme.halo_aware_indices, domain_LI)
    scheme.u0[mat_idx] = u[grid_idx]
  end

  prob = LinearProblem(scheme.A, scheme.b; u0=scheme.u0)
  LU = ilu(scheme.A; τ=0.1)
  @timeit "solve_step" sol = solve(prob, KrylovJL_GMRES(); Pl=LU)
  # @timeit "solve_step" sol = solve(prob)

  # update the solution to u in-place
  for (grid_idx, mat_idx) in zip(scheme.halo_aware_indices, domain_LI)
    u[grid_idx] = sol.u[mat_idx]
  end
  # return sol.resid, sol.iters, true

  return nothing
end

end
