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
using Printf

export ImplicitScheme, solve!, assemble_matrix!, initialize_coefficient_matrix
export DirichletBC, NeumannBC

struct ImplicitScheme{N,T,AA<:AbstractArray{T,N},SM,V,ST,PL,F,BC,CI1,CI2,SCL}
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
  domain_indices::CI1
  halo_aware_indices::CI2
  backend # GPU / CPU
  warmed_up::Vector{Bool}
  stencil_col_lookup::SCL
end

warmedup(scheme::ImplicitScheme) = scheme.warmed_up[1]
function warmedup!(scheme::ImplicitScheme)
  return scheme.warmed_up[1] = true
end

include("../averaging.jl")
include("../edge_terms.jl")
include("matrix_assembly.jl")
include("rhs_assembly.jl")
include("init_matrix.jl")

function ImplicitScheme(
  mesh,
  bcs;
  mean_func=arithmetic_mean,
  T=Float64,
  # solver=KrylovJL_GMRES(),
  backend=CPU(),
)
  celldims = cellsize_withhalo(mesh)

  len = prod(cellsize(mesh))

  halo_aware_CI = mesh.iterators.cell.domain
  domain_CI = CartesianIndices(size(halo_aware_CI))

  @assert length(domain_CI) == length(halo_aware_CI)

  A, stencil_col_lookup = initialize_coefficient_matrix(mesh, backend)
  Pl = preconditioner(A, backend)

  b = KernelAbstractions.zeros(backend, T, len)
  x = KernelAbstractions.zeros(backend, T, len)

  diffusivity = KernelAbstractions.zeros(backend, T, celldims)
  source_term = KernelAbstractions.zeros(backend, T, celldims)
  # memory = 50
  # solver = Krylov.DqgmresSolver(A, b, memory)
  solver = Krylov.GmresSolver(A, b)

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
    domain_CI,
    halo_aware_CI,
    backend,
    [false],
    stencil_col_lookup,
  )

  return implicit_solver
end

function solve!(
  scheme::ImplicitScheme{N,T},
  mesh::CurvilinearGrids.AbstractCurvilinearGrid,
  u,
  Δt;
  atol::T=√eps(T),
  rtol::T=√eps(T),
  maxiter=500,
  show_hist=true,
) where {N,T}

  #
  domain_LI = LinearIndices(scheme.domain_indices)

  A = scheme.A
  b = scheme.b
  x = scheme.solver.x

  # update the A matrix
  @timeit "assemble_matrix" assemble_matrix!(A, scheme, mesh, Δt)
  @timeit "assemble_rhs" assemble_rhs!(b, scheme, mesh, u, Δt)

  # precondition
  @timeit "ilu0! (preconditioner)" ilu0!(scheme.Pl, A)

  n = size(A, 1)
  precond_op = LinearOperator(
    eltype(A), n, n, false, false, (y, v) -> ldiv!(y, scheme.Pl, v)
  )

  if !warmedup(scheme)
    @timeit "linear solve (cold)" gmres!(
      scheme.solver, A, b; atol=atol, rtol=rtol, N=precond_op, history=true, itmax=maxiter
    )
    warmedup!(scheme)
  else
    @timeit "linear solve (warm)" gmres!(
      scheme.solver,
      A,
      b,
      x;
      atol=atol,
      rtol=rtol,
      N=precond_op,
      itmax=maxiter,
      history=true,
    )
  end

  L₂norm = last(scheme.solver.stats.residuals)
  niter = scheme.solver.stats.niter

  cutoff!(scheme.solver.x)

  # update u to the solution
  @views begin
    copyto!(u[scheme.halo_aware_indices], scheme.solver.x[domain_LI])
  end

  if show_hist
    @printf "\t Krylov stats: L₂norm: %.1e, iterations: %i\n" L₂norm niter
  end

  if !issolved(scheme.solver)
    @warn "The iterative solver didn't converge in the number of max iterations $(maxiter)"
  end

  return L₂norm, niter, issolved(scheme.solver)
end

# function working_solve!(
#   scheme::ImplicitScheme,
#   mesh::CurvilinearGrids.AbstractCurvilinearGrid,
#   u,
#   Δt;
#   maxiter=500,
#   show_hist=true,
# )
#   domain_LI = LinearIndices(scheme.domain_indices)

#   # update the A matrix
#   @timeit "assemble_matrix" assemble_matrix!(scheme, mesh, u, Δt)

#   # precondition
#   @timeit "ilu0! (preconditioner)" ilu0!(scheme.Pl, scheme.A)

#   n = size(scheme.A, 1)
#   precond_op = LinearOperator(
#     eltype(scheme.A), n, n, false, false, (y, v) -> ldiv!(y, scheme.Pl, v)
#   )

#   if !warmedup(scheme)
#     @timeit "dqgmres! (linear solve)" dqgmres!(
#       scheme.solver, scheme.A, scheme.b; N=precond_op, history=true, itmax=maxiter
#     )
#     warmedup!(scheme)
#   else
#     @timeit "dqgmres! (linear solve)" dqgmres!(
#       scheme.solver,
#       scheme.A,
#       scheme.b,
#       scheme.solver.x; # use last step as a starting point
#       N=precond_op,
#       itmax=maxiter,
#       history=true,
#     )
#   end

#   resids = last(scheme.solver.stats.residuals)
#   niter = scheme.solver.stats.niter

#   # update u to the solution
#   @views begin
#     copyto!(u[scheme.halo_aware_indices], scheme.solver.x[domain_LI])
#   end

#   # if show_hist
#   #   @printf "Convergence: %.1e, iterations: %i Δt: %.3e\n" resids niter Δt
#   # end

#   return resids, niter, issolved(scheme.solver)
# end

@inline function preconditioner(A, ::CPU, τ=0.1)
  return ILUZero.ilu0(A)
end

@inline function preconditioner(A, ::GPU, τ=0.1)
  return KrylovPreconditioners.kp_ilu0(A)
end

@inline cutoff(a) = (0.5(abs(a) + a)) * isfinite(a)

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
