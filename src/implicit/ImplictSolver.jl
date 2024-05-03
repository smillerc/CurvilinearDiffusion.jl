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

struct ImplicitScheme{N,T,AA<:AbstractArray{T,N},SM,V,ST,PL,F,BC,IT,SCL}
  A::SM # sparse matrix
  x::V # solution vector
  b::V # RHS vector
  solver::ST # linear solver, e.g. GMRES, CG, etc.
  Pl::PL # preconditioner
  α::AA # cell-centered diffusivity
  source_term::AA # cell-centered source term
  mean_func::F
  bcs::BC
  iterators::IT
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

function ImplicitScheme(mesh, bcs; mean_func=arithmetic_mean, T=Float64, backend=CPU())
  @assert mesh.nhalo >= 1 "The diffusion solver requires the mesh to have a halo region >= 1 cell wide"

  # The diffusion solver is currently set to use only 1 halo cell
  nhalo = 1

  # CartesianIndices used to iterate through the mesh; this is the same
  # size as the diffusion domain, but the indices are within the context
  # of the mesh (which will have the same or more halo cells, e.g. a hydro mesh with 6 halo cells)
  mesh_CI = expand(mesh.iterators.cell.domain, nhalo)

  # The CI CartesianIndices are used to iterate through the
  # entire problem, and the LI linear indices are to make it simple
  # to work with 1D indices for the A matrix and b rhs vector construction
  full_CI = CartesianIndices(size(mesh_CI))
  domain_CI = expand(full_CI, -nhalo)

  @assert length(full_CI) == length(mesh_CI)

  iterators = (
    mesh=mesh_CI, # used to access mesh quantities
    domain=(cartesian=domain_CI, linear=LinearIndices(domain_CI)),
    full=(cartesian=full_CI, linear=LinearIndices(full_CI)),
  )

  A, stencil_col_lookup = initialize_coefficient_matrix(iterators, mesh, backend)
  Pl = preconditioner(A, backend)

  b = KernelAbstractions.zeros(backend, T, length(full_CI))
  x = KernelAbstractions.zeros(backend, T, length(full_CI))

  diffusivity = KernelAbstractions.zeros(backend, T, size(full_CI))
  source_term = KernelAbstractions.zeros(backend, T, size(full_CI))

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
    iterators,
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

  @assert size(u) == size(mesh.iterators.cell.domain)

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
  domain_LI = scheme.iterators.domain.linear
  @views begin
    copyto!(
      u[mesh.iterators.cell.domain], #
      scheme.solver.x[domain_LI],    #
    )
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
