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

using KernelAbstractions
using LinearSolve
using ILUZero
using IncompleteLU
using LinearAlgebra
using OffsetArrays
using .Threads

export ImplicitScheme, solve!, assemble_matrix!

struct ImplicitScheme{N,LP,SM,V,ST,T,F,BC,CI1,CI2}
  prob::LP # linear problem (contains A, b, u0, solver...)
  A::SM # sparse matrix
  x::V # solution vector
  u0::V
  b::V # RHS vector
  solver::ST # linear solver, e.g. GMRES, CG, etc.
  # J::Array{T,N} # cell-centered Jacobian
  # metrics::Array{EM,N}
  α::Array{T,N} # cell-centered diffusivity
  source_term::Array{T,N} # cell-centered source term
  mean_func::F
  bcs::BC
  nhalo::Int
  conservative::Bool # uses the conservative form
  domain_indices::CI1
  halo_aware_indices::CI2
  backend
end

include("../averaging.jl")
include("../edge_terms.jl")
include("matrix_assembly.jl")

function ImplicitScheme(
  mesh::CurvilinearGrid2D,
  bcs;
  form=:conservative,
  mean_func=arithmetic_mean,
  T=Float64,
  solver=KrylovJL_GMRES(),
  backend=CPU(),
)
  celldims = cellsize_withhalo(mesh)

  ni, nj = cellsize(mesh)
  len = ni * nj

  halo_aware_CI = mesh.iterators.cell.domain
  domain_CI = CartesianIndices(size(halo_aware_CI))

  @assert length(domain_CI) == length(halo_aware_CI)

  conservative = form === :conservative

  A = init_A_matrix(mesh, backend)

  b = zeros(T, len)
  u0 = zeros(T, len) # initial guess for iterative solvers
  x = zeros(T, len)

  diffusivity = zeros(T, celldims)
  source_term = zeros(T, celldims)

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
    diffusivity,
    source_term,
    mean_func,
    bcs,
    mesh.nhalo,
    conservative,
    domain_CI,
    halo_aware_CI,
    backend,
  )

  return implicit_solver
end

function init_A_matrix(mesh::CurvilinearGrid2D, ::CPU)
  ni, nj = cellsize(mesh)
  len = ni * nj

  #! format: off
  A = spdiagm(
    -ni - 1 => zeros(len - ni - 1), # (i-1, j-1)
    -ni     => zeros(len - ni),     # (i  , j-1)
    -ni + 1 => zeros(len - ni + 1), # (i+1, j-1)
    -1      => zeros(len - 1),      # (i-1, j  )
    0       => ones(len),           # (i  , j  )
    1       => zeros(len - 1),      # (i+1, j  )
    ni - 1  => zeros(len - ni + 1), # (i-1, j+1)
    ni      => zeros(len - ni),     # (i  , j+1)
    ni + 1  => zeros(len - ni - 1), # (i+1, j+1)
  )
  #! format: on
  return A
end

function init_A_matrix(mesh::CurvilinearGrid2D, ::GPU)
  return CuSparseMatrixCSR(init_A_matrix(mesh, CPU()))
end

function init_A_matrix(mesh::CurvilinearGrid1D, ::CPU)
  len, = cellsize(mesh)
  A = Tridiagonal(zeros(len - 1), ones(len), zeros(len - 1))
  return A
end

function solve!(
  scheme::ImplicitScheme, mesh::CurvilinearGrids.AbstractCurvilinearGrid, u, Δt
)
  domain_LI = LinearIndices(scheme.domain_indices)

  # update the A matrix
  @timeit "assemble_matrix" assemble_matrix!(scheme, mesh, u, Δt)

  # @timeit "solve_step" LinearSolve.solve!(scheme.prob)

  # for (grid_idx, mat_idx) in zip(scheme.halo_aware_indices, domain_LI)
  #   u[grid_idx] = scheme.prob.u[mat_idx]
  # end

  # use the current solution as a guess for the next
  for (grid_idx, mat_idx) in zip(scheme.halo_aware_indices, domain_LI)
    scheme.u0[mat_idx] = u[grid_idx]
  end

  prob = LinearProblem(scheme.A, scheme.b; u0=scheme.u0)
  @timeit "preconditioner" Pl = preconditioner(scheme.backend, scheme.A)
  @timeit "solve" sol = solve(prob, KrylovJL_GMRES(); Pl=Pl)
  # @timeit "solve_step" sol = solve(prob; Pl=LU)

  # update the solution to u in-place
  for (grid_idx, mat_idx) in zip(scheme.halo_aware_indices, domain_LI)
    u[grid_idx] = sol.u[mat_idx]
  end
  # return sol.resid, sol.iters, true

  return nothing
end

@inline function preconditioner(::CPU, A, τ=0.1)
  return IncompleteLU.ilu(A; τ=τ)
end

@inline function preconditioner(::GPU, A, τ=0.1)
  return KrylovPreconditioners.kp_ilu0(A)
end

end
