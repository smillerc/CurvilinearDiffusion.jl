module ImplicitSchemeType

using CartesianDomains
using CurvilinearGrids
using ILUZero
using KernelAbstractions
using Krylov
using KrylovPreconditioners
using LinearAlgebra
using LinearOperators
using LinearSolve
using Printf
using SparseArrays
using SparseMatricesCSR
using StaticArrays
using TimerOutputs
using UnPack
using Pardiso

using ..BoundaryConditions
using ..BoundaryConditions: bc_rhs_coefficient
using ..TimeStepControl: next_dt

export ImplicitScheme, solve!, assemble!, initialize_coefficient_matrix
export DirichletBC, NeumannBC, PeriodicBC, applybc!, applybcs!, check_diffusivity_validity

struct ImplicitScheme{N,T,AA<:AbstractArray{T,N},ST,F,BC,IT,L,BE}
  linear_problem::ST # linear solver, e.g. GMRES, CG, etc.
  # uⁿ⁺¹::AA
  α::AA # cell-centered diffusivity
  source_term::AA # cell-centered source term
  mean_func::F
  bcs::BC
  iterators::IT
  limits::L
  backend::BE # GPU / CPU
  warmed_up::Vector{Bool}
  direct_solve::Bool
end

warmedup(scheme::ImplicitScheme) = scheme.warmed_up[1]
function warmup!(scheme::ImplicitScheme)
  return scheme.warmed_up[1] = true
end

include("../averaging.jl")
include("../edge_terms.jl")
include("matrix_assembly.jl")
include("init_matrix.jl")

function ImplicitScheme(
  mesh,
  bcs;
  direct_solve=false,
  face_conductivity::Symbol=:harmonic,
  T=Float64,
  backend=CPU(),
)
  @assert mesh.nhalo >= 1 "The diffusion solver requires the mesh to have a halo region >= 1 cell wide"

  if face_conductivity === :harmonic
    mean_func = harmonic_mean
    @info "Using harmonic mean for face conductivity averaging"
  else
    @info "Using arithmetic mean for face conductivity averaging"
    mean_func = arithmetic_mean
  end
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
  @info "Initializing the sparse A coefficient matrix"
  A = initialize_coefficient_matrix(iterators, mesh, bcs, backend)

  b = KernelAbstractions.zeros(backend, T, length(full_CI))
  diffusivity = KernelAbstractions.zeros(backend, T, size(full_CI))
  uⁿ⁺¹ = KernelAbstractions.zeros(backend, T, size(full_CI))

  source_term = KernelAbstractions.zeros(backend, T, size(full_CI))

  if direct_solve
    # algorithm = UMFPACKFactorization(; reuse_symbolic=true, check_pattern=true)
    # algorithm = KLUFactorization(; reuse_symbolic=true, check_pattern=true)
    # algorithm = MKLPardisoFactorize()
    algorithm = MKLPardisoIterate()

    linear_problem = init(LinearProblem(A, b), algorithm)
  else
    # Pr, _ldiv = preconditioner(A, backend)
    # algorithm = KrylovJL_GMRES(; history=true, ldiv=_ldiv) # krylov solver

    solver = GmresSolver(A, b)

    F = ilu0(A)
    linear_problem = (; solver, A, b, precon=F)
  end

  implicit_solver = ImplicitScheme(
    linear_problem,
    # uⁿ⁺¹,
    diffusivity,
    source_term,
    mean_func,
    bcs,
    iterators,
    _limits,
    backend,
    [false],
    direct_solve,
  )

  @info "Initialization finished"
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

function solve!(scheme::ImplicitScheme, mesh, u, Δt; kwargs...)
  if scheme.direct_solve
    _direct_solve!(scheme, mesh, u, Δt; kwargs...)
  else
    _iterative_solve!(scheme, mesh, u, Δt; kwargs...)
  end
end

function _direct_solve!(
  scheme::ImplicitScheme{N,T},
  mesh::CurvilinearGrids.AbstractCurvilinearGrid,
  u,
  Δt;
  show_convergence=true,
  cutoff=true,
  kwargs...,
) where {N,T}

  #
  domain_u = @views u[scheme.iterators.mesh]

  @assert size(u) == size(mesh.iterators.cell.full)

  # update the A matrix and b vector; A is a separate argument
  # so we can dispatch on type for GPU vs CPU assembly
  @timeit "assembly" assemble!(scheme.linear_problem.A, u, scheme, mesh, Δt)
  KernelAbstractions.synchronize(scheme.backend)

  # For the direct solve, we want to re-use the symbolic factorization (sparsity pattern)
  # but update the A matrix (which we did above via assemble!(...))
  # Setting isfresh=true will tell the direct solver that the 
  # A matrix has been changed. For iterative Krylov solvers, we don't need to 
  # do this
  scheme.linear_problem.isfresh = true

  if !warmedup(scheme)
    if scheme.direct_solve
      @info "Performing the first (cold) factorization (if direct) and solve, this will be re-used in subsequent solves"
    end

    @timeit "linear solve (cold)" LinearSolve.solve!(scheme.linear_problem; kwargs...)
    warmup!(scheme)
  else
    @timeit "linear solve (warm)" LinearSolve.solve!(scheme.linear_problem; kwargs...)
  end

  # Apply a cutoff function to remove negative u values
  if cutoff
    cutoff!(scheme.linear_problem.u)
  end

  @timeit "next_dt" begin
    next_Δt = next_dt(scheme.linear_problem.u, domain_u, Δt; kwargs...)
  end

  copyto!(domain_u, scheme.linear_problem.u) # update solution

  if !scheme.direct_solve
    # if !scheme.linear_problem.cacheval.stats.solved
    #   @show scheme.linear_problem.cacheval.stats
    #   error("The solver didn't converge")
    # end

    L₂norm = last(scheme.linear_problem.cacheval.stats.residuals)
    niter = scheme.linear_problem.cacheval.stats.niter

    if show_convergence
      @printf "\tKrylov stats: L₂: %.1e, iterations: %i\n" L₂norm niter
    end

    # return L₂norm, niter, true
    return L₂norm, next_Δt
  else
    return -Inf, next_Δt
    # return -Inf, 1, true
  end
end

function _iterative_solve!(
  scheme::ImplicitScheme{N,T},
  mesh::CurvilinearGrids.AbstractCurvilinearGrid,
  u,
  Δt;
  show_convergence=true,
  cutoff=true,
  kwargs...,
) where {N,T}

  #
  domain_u = @views u[scheme.iterators.mesh]

  @assert size(u) == size(mesh.iterators.cell.full)

  # update the A matrix and b vector; A is a separate argument
  # so we can dispatch on type for GPU vs CPU assembly
  @timeit "assembly" assemble!(scheme.linear_problem.A, u, scheme, mesh, Δt)
  KernelAbstractions.synchronize(scheme.backend)

  n = size(scheme.linear_problem.A, 1)
  F = scheme.linear_problem.precon
  @timeit "ilu0!" ilu0!(F, scheme.linear_problem.A)

  opM = LinearOperator(
    Float64, n, n, false, false, (y, v) -> forward_substitution!(y, F, v)
  )
  opN = LinearOperator(
    Float64, n, n, false, false, (y, v) -> backward_substitution!(y, F, v)
  )

  @timeit "linear solve!" Krylov.solve!(
    scheme.linear_problem.solver,
    scheme.linear_problem.A,
    scheme.linear_problem.b;
    history=true,
    M=opM,
    N=opN,
  )

  # Apply a cutoff function to remove negative u values
  if cutoff
    cutoff!(scheme.linear_problem.solver.x)
  end

  @timeit "next_dt" begin
    next_Δt = next_dt(scheme.linear_problem.solver.x, domain_u, Δt; kwargs...)
  end

  copyto!(domain_u, scheme.linear_problem.solver.x) # update solution

  niter = scheme.linear_problem.solver.stats.niter
  L₂norm = last(scheme.linear_problem.solver.stats.residuals)

  if show_convergence
    @printf "\tKrylov stats: L₂: %.1e, iterations: %i\n" L₂norm niter
  end

  return L₂norm, next_Δt
end

preconditioner(A, ::CPU) = ILUZero.ilu0

function preconditioner(A, backend::GPU, τ=0.1)
  p = KrylovPreconditioners.kp_ilu0(A)
  # p = KrylovPreconditioners.BlockJacobiPreconditioner(A, backend)
  return p
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

  corners = domain_corners(scheme.iterators.full.cartesian)
  _kernel(scheme.backend)(scheme.α, corners; ndrange=size(scheme.α))
  return nothing
end

function domain_corners(::CartesianIndices{1})
  return (nothing,)
end

function domain_corners(CI::CartesianIndices{2})
  @views begin
    corners = (CI[begin, begin], CI[begin, end], CI[end, begin], CI[end, end])
  end
  return corners
end

function domain_corners(CI::CartesianIndices{3})
  @views begin
    corners = (
      CI[begin, begin, begin],
      CI[end, begin, begin],
      CI[begin, end, begin],
      CI[end, end, begin],
      CI[begin, begin, end],
      CI[end, begin, end],
      CI[begin, end, end],
      CI[end, end, end],
    )
  end
  return corners
end

end
