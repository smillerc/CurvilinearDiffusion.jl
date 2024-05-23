module ImplicitSchemeType

using CartesianDomains
using CurvilinearGrids
using ILUZero
using IncompleteLU
using KernelAbstractions
using Krylov
using KrylovPreconditioners
using LinearAlgebra
using LinearSolve
using Printf
using SparseArrays
using SparseMatricesCSR
using StaticArrays
using TimerOutputs
using UnPack

export ImplicitScheme, solve!, assemble!, initialize_coefficient_matrix
export DirichletBC, NeumannBC, PeriodicBC, applybc!, applybcs!, check_diffusivity_validity

struct ImplicitScheme{N,T,AA<:AbstractArray{T,N},ST,F,BC,IT,L}
  linear_problem::ST # linear solver, e.g. GMRES, CG, etc.
  α::AA # cell-centered diffusivity
  source_term::AA # cell-centered source term
  mean_func::F
  bcs::BC
  iterators::IT
  limits::L
  backend # GPU / CPU
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
  source_term = KernelAbstractions.zeros(backend, T, size(full_CI))

  # solver = Krylov.GmresSolver(A, b)
  if direct_solve
    algorithm = UMFPACKFactorization()
    linear_problem = init(LinearProblem(A, b), algorithm; verbose=true)
    # linear_problem = (; A, b, alg=UMFPACKFactorization())
  else
    Pl, _ldiv = preconditioner(A, backend)
    algorithm = KrylovJL_GMRES(; history=true, ldiv=_ldiv) # krylov solver
    linear_problem = init(LinearProblem(A, b), algorithm; Pl=Pl)
    # linear_problem = LinearProblem(A, b)

    # algorithm = HYPREAlgorithm(HYPRE.GMRES)
    # Pl = HYPRE.BoomerAMG
    # algorithm = KrylovJL_GMRES(; history=true) # krylov solver
    # linear_problem = init(LinearProblem(A, b), algorithm;)
  end

  implicit_solver = ImplicitScheme(
    linear_problem,
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

function solve!(
  scheme::ImplicitScheme{N,T},
  mesh::CurvilinearGrids.AbstractCurvilinearGrid,
  u,
  Δt;
  atol::T=√eps(T),
  rtol::T=√eps(T),
  maxiter=200,
  show_convergence=true,
  cutoff=true,
) where {N,T}
  #

  domain_u = @view u[scheme.iterators.mesh]

  @assert size(u) == size(mesh.iterators.cell.full)

  # update the A matrix and b vector; A is a separate argument
  # so we can dispatch on type for GPU vs CPU assembly
  @timeit "assembly" assemble!(scheme.linear_problem.A, u, scheme, mesh, Δt)
  KernelAbstractions.synchronize(scheme.backend)

  # prob = LinearProblem(scheme.linear_problem.A, scheme.linear_problem.b)
  # sol = solve(prob)
  # if cutoff
  #   cutoff!(sol.u)
  # end
  # copyto!(domain_u, sol.u) # update solution

  scheme.linear_problem.isfresh = true
  if !warmedup(scheme)
    if scheme.direct_solve
      @info "Performing the first (cold) factorization (if direct) and solve, this will be re-used in subsequent solves"
    end

    @timeit "linear solve (cold)" LinearSolve.solve!(scheme.linear_problem; alias_A=true)
    warmup!(scheme)
  else
    @timeit "linear solve (warm)" LinearSolve.solve!(scheme.linear_problem)
  end

  if cutoff
    cutoff!(scheme.linear_problem.u)
  end
  copyto!(domain_u, scheme.linear_problem.u) # update solution

  if !scheme.direct_solve
    L₂norm = last(scheme.linear_problem.cacheval.stats.residuals)
    niter = scheme.linear_problem.cacheval.stats.niter
    is_solved = scheme.linear_problem.cacheval.stats.solved

    # if !is_solved
    #   @show scheme.linear_problem.cacheval.stats
    #   error(
    #     "The iterative solver didn't converge in the number of max iterations $(maxiter)"
    #   )
    # end

    if show_convergence
      @printf "\tKrylov stats: L₂: %.1e, iterations: %i\n" L₂norm niter
    end

    return L₂norm, niter, true # issolved(scheme.linear_problem)
  else
    return -Inf, 1, true
  end
end

@inline function preconditioner(A, ::CPU, τ=0.1)
  # p = ILUZero.ilu0(A)
  p = IncompleteLU.ilu(A)
  _ldiv = true
  return p, _ldiv
end

@inline function preconditioner(A, backend::GPU, τ=0.1)
  p = KrylovPreconditioners.kp_ilu0(A)
  _ldiv = true
  # nblocks = 4

  # p = KrylovPreconditioners.BlockJacobiPreconditioner(A, nblocks, backend)
  # _ldiv = false

  return p, _ldiv
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
