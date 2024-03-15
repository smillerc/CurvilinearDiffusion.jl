module ImplicitSchemeType

using .Threads
using CurvilinearGrids
using ILUZero
using KernelAbstractions
using Krylov
using KrylovPreconditioners
using LinearAlgebra
using LinearOperators
using LinearSolve
using OffsetArrays
using SparseArrays
using StaticArrays
using TimerOutputs
using UnPack

export ImplicitScheme, solve!, assemble_matrix!

struct ImplicitScheme{N,T,AA<:AbstractArray{T,N},SM,V,ST,PL,F,BC,CI1,CI2}
  A::SM # sparse matrix
  x::V # solution vector
  b::V # RHS vector
  solver::ST # linear solver, e.g. GMRES, CG, etc.
  Pl::PL # preconditioner
  α::AA # cell-centered diffusivity
  source_term::AA # cell-centered source term
  mean_func::F
  bcs::BC
  nhalo::Int
  conservative::Bool # uses the conservative form
  domain_indices::CI1
  halo_aware_indices::CI2
  backend # GPU / CPU
  warmed_up::Vector{Bool}
end

warmedup(scheme::ImplicitScheme) = scheme.warmed_up[1]
function warmedup!(scheme::ImplicitScheme)
  return scheme.warmed_up[1] = true
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
  # solver=KrylovJL_GMRES(),
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
  Pl = preconditioner(A, backend)

  b = KernelAbstractions.zeros(backend, T, len)
  x = KernelAbstractions.zeros(backend, T, len)

  diffusivity = KernelAbstractions.zeros(backend, T, celldims)
  source_term = KernelAbstractions.zeros(backend, T, celldims)
  memory = 50
  solver = Krylov.DqgmresSolver(A, b, memory)
  # solver = Krylov.GmresSolver(A, b)

  implicit_solver = ImplicitScheme(
    A,
    x,
    b,
    solver,
    Pl,
    diffusivity,
    source_term,
    mean_func,
    bcs,
    mesh.nhalo,
    conservative,
    domain_CI,
    halo_aware_CI,
    backend,
    [false],
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

  # kv = (
  #   -ni - 1 => zeros(len - ni - 1), # (i-1, j-1)
  #   -ni => zeros(len - ni),     # (i  , j-1)
  #   -ni + 1 => zeros(len - ni + 1), # (i+1, j-1)
  #   -1 => zeros(len - 1),      # (i-1, j  )
  #   0 => ones(len),           # (i  , j  )
  #   1 => zeros(len - 1),      # (i+1, j  )
  #   ni - 1 => zeros(len - ni + 1), # (i-1, j+1)
  #   ni => zeros(len - ni),     # (i  , j+1)
  #   ni + 1 => zeros(len - ni - 1), # (i+1, j+1)
  # )

  # I, J, V, mmax, nmax = SparseArrays.spdiagm_internal(kv...)
  # B = sparsecsr(I, J, V, mmax, nmax)

  return A
end

function init_A_matrix(mesh::CurvilinearGrid1D, ::CPU)
  len, = cellsize(mesh)
  A = Tridiagonal(zeros(len - 1), ones(len), zeros(len - 1))
  return A
end

function solve!(
  scheme::ImplicitScheme, mesh::CurvilinearGrids.AbstractCurvilinearGrid, u, Δt; maxiter=Inf
)
  domain_LI = LinearIndices(scheme.domain_indices)

  # update the A matrix
  @timeit "assemble_matrix" assemble_matrix!(scheme, mesh, u, Δt)

  # precondition
  @timeit "ilu0! (preconditioner)" ilu0!(scheme.Pl, scheme.A)

  n = size(scheme.A, 1)
  precond_op = LinearOperator(
    eltype(scheme.A), n, n, false, false, (y, v) -> ldiv!(y, scheme.Pl, v)
  )

  if !warmedup(scheme)
    @timeit "dqgmres! (linear solve)" dqgmres!(
      scheme.solver, scheme.A, scheme.b; N=precond_op, history=true, itmax=maxiter
    )
    warmedup!(scheme)
  else
    @timeit "dqgmres! (linear solve)" dqgmres!(
      scheme.solver,
      scheme.A,
      scheme.b,
      scheme.solver.x; # use last step as a starting point
      N=precond_op,
      itmax=maxiter,
      history=true,
    )
  end

  resids = last(scheme.solver.stats.residuals)
  niter = scheme.solver.stats.niter

  # update u to the solution
  @views begin
    copyto!(u[scheme.halo_aware_indices], scheme.solver.x[domain_LI])
  end
  @show resids, niter, issolved(scheme.solver)
  return resids, niter, issolved(scheme.solver)
end

@inline function preconditioner(A, ::CPU, τ=0.1)
  return ILUZero.ilu0(A)
end

@inline function preconditioner(A, ::GPU, τ=0.1)
  return KrylovPreconditioners.kp_ilu0(A)
end

end
