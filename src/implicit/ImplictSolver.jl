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
export DirichletBC, NeumannBC, applybc!, applybcs!

struct ImplicitScheme{N,T,AA<:AbstractArray{T,N},SM,V,ST,PL,F,BC,IT,L}
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
  limits::L
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
include("init_matrix.jl")

function ImplicitScheme(mesh, bcs; mean_func=arithmetic_mean, T=Float64, backend=CPU())
  @assert mesh.nhalo >= 1 "The diffusion solver requires the mesh to have a halo region >= 1 cell wide"

  # The diffusion solver is currently set to use only 1 halo cell
  nhalo = 1

  # CartesianIndices used to iterate through the mesh; this is the same
  # size as the diffusion domain, but the indices are within the context
  # of the mesh (which will have the same or more halo cells, e.g. a hydro mesh with 6 halo cells)
  mesh_CI = expand(mesh.iterators.cell.domain, nhalo)

  # The diffusion domain is nested within the mesh extents, since
  # the mesh can have a larger halo/ghost region;
  #
  #   +--------------------------------------+
  #   |                                      |
  #   |   +------------------------------+   |
  #   |   |                              |   |
  #   |   |   +----------------------+   |   |
  #   |   |   |                      |   |   |
  #   |   |   |                      |   |   |
  #   |   |   |                      |   |   |
  #   |   |   |                      |   |   |
  #   |   |   |(1)                   |   |   |
  #   |   |   +----------------------+   |   |
  #   |   |(2)                           |   |
  #   |   +------------------------------+   |
  #   |(3)                                   |
  #   +--------------------------------------+

  # (1) is the "domain" region of both the mesh and the diffusion problem
  # (2) is the full extent of the diffusion problem (region 1 + 1 halo cell)
  # (3) is the full extent of the mesh (region 1 + n halo cells)

  # The regions (2) and (3) will be the same size if the mesh has 1 halo cell
  # The linear problem Ax=b that this scheme solves is done within region (2),
  # and boundary conditions are handled via ghost/halo cells.

  # The CI CartesianIndices are used to iterate through the
  # entire problem, and the LI linear indices are to make it simple
  # to work with 1D indices for the A matrix and b rhs vector construction
  full_CI = CartesianIndices(size(mesh_CI))
  domain_CI = expand(full_CI, -nhalo)

  @assert length(full_CI) == length(mesh_CI)

  iterators = (
    domain=( # region 1, but within the context of region 2
      cartesian=domain_CI,
      linear=LinearIndices(domain_CI),
    ),
    full=( # region 2
      cartesian=full_CI,
      linear=LinearIndices(full_CI),
    ),
    mesh=mesh_CI, # region 2, but within the context of region 3
  )

  _limits = limits(full_CI)
  A = initialize_coefficient_matrix(iterators, mesh, bcs, backend)
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
    _limits,
    backend,
    [false],
  )

  return implicit_solver
end

function limits(CI::CartesianIndices{2})
  lo = first.(CI.indices)
  hi = last.(CI.indices)
  return (ilo=lo[1], jlo=lo[2], ihi=hi[1], jhi=hi[2])
end

function limits(CI::CartesianIndices{3})
  lo = first.(CI.indices)
  hi = last.(CI.indices)
  return (ilo=lo[1], jlo=lo[2], klo=lo[3], ihi=hi[1], jhi=hi[2], khi=hi[3])
end

function solve!(
  scheme::ImplicitScheme{N,T},
  mesh::CurvilinearGrids.AbstractCurvilinearGrid,
  u,
  Δt;
  atol::T=√eps(T),
  rtol::T=√eps(T),
  maxiter=200,
  show_hist=true,
) where {N,T}
  #

  @assert size(u) == size(mesh.iterators.cell.full)

  A = scheme.A
  b = scheme.b
  x = scheme.solver.x

  # update the A matrix
  @timeit "assembly" assemble!(A, u, scheme, mesh, Δt)
  # @timeit "assemble_rhs" assemble_rhs!(b, scheme, mesh, u, Δt)

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
  domain_LI = scheme.iterators.full.linear
  @views begin
    copyto!(
      u[scheme.iterators.mesh], #
      scheme.solver.x[domain_LI],    #
    )
  end

  if show_hist
    @printf "\t Krylov stats: L₂norm: %.1e, iterations: %i\n" L₂norm niter
  end

  if !issolved(scheme.solver)
    @show scheme.solver.stats
    error("The iterative solver didn't converge in the number of max iterations $(maxiter)")
  end

  return L₂norm, niter, issolved(scheme.solver)
end

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
