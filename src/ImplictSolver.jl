module ImplicitSchemeType

using LinearSolve
using CurvilinearGrids
using Polyester, StaticArrays
using UnPack

using SparseArrays
using StaticArrays
# using AlgebraicMultigrid
# using Krylov
# using LinearSolve
using ILUZero
using IncompleteLU
using LinearAlgebra
using OffsetArrays

export ImplicitScheme, solve!, assemble_matrix!

struct ImplicitScheme{LP,ST,T,N,EM,F,BC,CI1,CI2}
  prob::LP # linear problem (contains A, b, u0, solver...)
  # A::SM # sparse matrix
  # x::V # solution vector
  # b::V # RHS vector
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
    metric_type = typeof(_conservative_metrics_2d(mesh, 1, 1))
  else
    conservative = false
    metric_type = typeof(_non_conservative_metrics_2d(mesh, 1, 1))
  end

  b = zeros(T, len)
  u0 = zeros(T, len) # initial guess for iterative solvers
  x = zeros(T, len)
  edge_metrics = Array{metric_type,2}(undef, celldims)

  diffusivity = zeros(T, celldims)
  source_term = zeros(T, celldims)
  J = zeros(T, celldims)

  prob = LinearProblem(A, b)

  linear_problem = init(prob; alias_A=false, alias_b=false)
  # linear_problem = init(prob, KrylovJL_GMRES())
  # linear_problem = LinearProblem(A, b; u0=u0)

  implicit_solver = ImplicitScheme(
    linear_problem,
    # A,
    # x,
    # b,
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

function solve!(scheme::ImplicitScheme, mesh, u, Δt)

  # update the A matrix
  assemble_matrix!(scheme, u, Δt)

  domain_LI = LinearIndices(scheme.domain_indices)

  # solve it !
  # prob = LinearProblem(solver.A, solver.b)
  # Pl = ilu0(A)

  # LU = ilu(scheme.prob.A; τ=0.1)
  # scheme.prob.Pl = LU
  LinearSolve.solve!(scheme.prob)

  for (grid_idx, mat_idx) in zip(scheme.halo_aware_indices, domain_LI)
    u[grid_idx] = scheme.prob.u[mat_idx]
  end

  # use the current solution as a guess for the next
  # for (grid_idx, mat_idx) in zip(scheme.halo_aware_indices, domain_LI)
  #   scheme.prob.u0[mat_idx] = u[grid_idx]
  # end
  # LU = ilu(scheme.prob.A; τ=0.1)
  # sol = solve(scheme.prob, KrylovJL_GMRES(); Pl=LU, verbose=false)

  # # update the solution to u in-place
  # for (grid_idx, mat_idx) in zip(scheme.halo_aware_indices, domain_LI)
  #   u[grid_idx] = sol.u[mat_idx]
  # end
  # return sol.resid, sol.iters, true

end

end
