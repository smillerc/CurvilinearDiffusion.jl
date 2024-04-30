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
using SparseMatricesCSR
using BandedMatrices
using IncompleteLU
using CUDA
using CUDSS
using CUDA.CUSPARSE: CuSparseMatrixCSR

export ImplicitScheme, solve!, assemble_matrix!, initialize_coefficient_matrix

# struct ImplicitScheme{N,T,AA<:AbstractArray{T,N},SM,SM2,V,ST,PL,F,BC,CI1,CI2}
struct ImplicitScheme{N,T,BE,AA<:AbstractArray{T,N},ST,F,BC,CI1,CI2,SCL}
  linear_problem::ST # linear solver
  α::AA # cell-centered diffusivity
  source_term::AA # cell-centered source term
  mean_func::F
  bcs::BC
  nhalo::Int
  domain_indices::CI1
  halo_aware_indices::CI2
  backend::BE # GPU / CPU
  warmed_up::Vector{Bool}
  direct_solve::Bool
  stencil_col_lookup::SCL
end

warmedup(scheme::ImplicitScheme) = scheme.warmed_up[1]
function warmup!(scheme::ImplicitScheme)
  return scheme.warmed_up[1] = true
end

include("../averaging.jl")
# include("../edge_terms.jl")
include("matrix_assembly.jl")
include("init_matrix.jl")
include("rhs_assembly.jl")

function ImplicitScheme(
  mesh::CurvilinearGrid2D,
  bcs;
  direct_solve=true,
  mean_func=arithmetic_mean,
  T=Float64,
  backend=CPU(),
)
  @info "Initializing ImplicitScheme"

  celldims = cellsize_withhalo(mesh)

  ni, nj = cellsize(mesh)
  len = ni * nj

  halo_aware_CI = mesh.iterators.cell.domain
  domain_CI = CartesianIndices(size(halo_aware_CI))

  @assert length(domain_CI) == length(halo_aware_CI)

  A, stencil_col_lookup = initialize_coefficient_matrix(mesh, backend)

  b = KernelAbstractions.zeros(backend, T, len)
  x = KernelAbstractions.zeros(backend, T, len)

  diffusivity = KernelAbstractions.zeros(backend, T, celldims)
  source_term = KernelAbstractions.zeros(backend, T, celldims)

  if direct_solve
    if backend isa CUDABackend
      linear_problem = (A=A, b=b, x=x, solver=CudssSolver(A, "G", 'F'))
    else
      linear_problem = init(LinearProblem(A, b))
      # linear_problem = LinearProblem(A, b)
    end

  else
    Pl = preconditioner(A, backend)
    linear_problem = (A=A, b=b, x=x, solver=Krylov.GmresSolver(A, b), Pl=Pl)
    # # Pl = kp_ilu0(A)
    # linear_problem = init(
    #   LinearProblem(A, b),
    #   # KrylovJL_GMRES(; history=true, N=Pl, ldiv=false); # krylov solver
    #   KrylovJL_GMRES(; history=true); # krylov solver
    #   # Pl=Pl,
    # )

  end

  implicit_solver = ImplicitScheme(
    linear_problem,
    diffusivity,
    source_term,
    mean_func,
    bcs,
    mesh.nhalo,
    domain_CI,
    halo_aware_CI,
    backend,
    [false],
    direct_solve,
    stencil_col_lookup,
  )

  @info "Done"
  return implicit_solver
end

function initialize_coefficient_matrix(mesh::CurvilinearGrid2D, ::CPU)
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
  # A = sparsecsr(I, J, V, mmax, nmax)

  return A, nothing
end

# function initialize_coefficient_matrix(mesh::CurvilinearGrid1D, ::CPU)
#   len, = cellsize(mesh)
#   A = Tridiagonal(zeros(len - 1), ones(len), zeros(len - 1))
#   return A
# end

function solve!(
  scheme::ImplicitScheme,
  mesh::CurvilinearGrids.AbstractCurvilinearGrid,
  u,
  Δt;
  maxiter=500,
  show_hist=true,
)
  if scheme.backend isa CPU
    if scheme.direct_solve
      _cpu_direct_solve!(scheme, mesh, u, Δt)
    else
      _cpu_iterative_solve!(scheme, mesh, u, Δt; maxiter=maxiter, show_hist=show_hist)
    end
  else
    if scheme.direct_solve
      _gpu_direct_solve!(scheme, mesh, u, Δt)
    else
      _gpu_iterative_solve!(scheme, mesh, u, Δt)
    end
  end

  return nothing
end

function _cpu_iterative_solve!(
  scheme::ImplicitScheme,
  mesh::CurvilinearGrids.AbstractCurvilinearGrid,
  u,
  Δt;
  maxiter=500,
  show_hist=true,
)
  A = scheme.linear_problem.A
  b = scheme.linear_problem.b
  x = scheme.linear_problem.x
  Pl = scheme.linear_problem.Pl
  solver = scheme.linear_problem.solver
  domain_LI = LinearIndices(scheme.domain_indices)

  @timeit "assemble_rhs!" assemble_rhs!(b, scheme, u, Δt)
  @timeit "assemble_matrix!" assemble_matrix!(scheme, A, mesh, Δt)

  # # error("done")
  # if !warmedup(scheme)
  #   @info "Performing the first factorization and solve (cold), after this the factorization will be re-used"
  #   @timeit "linear solve (cold)" LinearSolve.solve!(scheme.linear_problem)
  #   warmup!(scheme)
  # else
  #   @timeit "linear solve (warm)" LinearSolve.solve!(scheme.linear_problem)
  # end

  # precondition
  @timeit "ilu0! (preconditioner)" ilu0!(Pl, A)

  n = size(A, 1)
  precond_op = LinearOperator(eltype(A), n, n, false, false, (y, v) -> ldiv!(y, Pl, v))

  if !warmedup(scheme)
    @timeit "linear solve (cold)" gmres!(
      solver, A, b; N=precond_op, history=true, itmax=maxiter
    )
    warmup!(scheme)
  else
    @timeit "linear solve (warm)" gmres!(
      solver,
      A,
      b,
      x; # use last step as a starting point
      N=precond_op,
      itmax=maxiter,
      history=true,
    )
  end

  @views begin
    copyto!(u[scheme.halo_aware_indices], solver.x[domain_LI])
    # copyto!(u[scheme.halo_aware_indices], scheme.linear_problem.u[domain_LI])
  end

  # residual = last(scheme.linear_problem.cacheval.stats.residuals)
  # niter = scheme.linear_problem.cacheval.stats.niter
  residual = last(solver.stats.residuals)
  niter = solver.stats.niter
  if show_hist
    @printf "\tConvergence L₂norm : %.1e, solver iterations: %i\n" residual niter
  end
end

function _cpu_direct_solve!(
  scheme::ImplicitScheme, mesh::CurvilinearGrids.AbstractCurvilinearGrid, u, Δt;
)
  domain_LI = LinearIndices(scheme.domain_indices)

  @timeit "assemble_rhs!" assemble_rhs!(scheme.linear_problem.b, scheme, u, Δt)
  @timeit "assemble_matrix!" assemble_matrix!(scheme, scheme.linear_problem.A, mesh, Δt)

  # # error("done")
  if !warmedup(scheme)
    @info "Performing the first factorization and solve (cold), after this the factorization will be re-used"
    @timeit "linear solve (cold)" LinearSolve.solve!(scheme.linear_problem)
    warmup!(scheme)
  else
    @timeit "linear solve (warm)" LinearSolve.solve!(scheme.linear_problem)
  end

  # update u to the solution
  @views begin
    copyto!(u[scheme.halo_aware_indices], scheme.linear_problem.u[domain_LI])
  end

  return nothing
end

function _gpu_direct_solve!(
  scheme::ImplicitScheme, mesh::CurvilinearGrids.AbstractCurvilinearGrid, u, Δt;
)
  @timeit "assemble_rhs!" assemble_rhs!(scheme, u, Δt)
  @timeit "assemble_matrix!" assemble_matrix!(scheme, scheme.linear_problem.A, mesh, Δt)
  KernelAbstractions.synchronize(scheme.backend)

  @timeit "cudss_set" cudss_set(scheme.linear_problem.solver, scheme.linear_problem.A)
  if !warmedup(scheme)
    @info "Performing the first factorization and solve (cold), after this the factorization will be re-used"

    @timeit "analyze (cold)" cudss(
      "analysis",
      scheme.linear_problem.solver,
      scheme.linear_problem.x,
      scheme.linear_problem.b,
    )
    @timeit "factorize (cold)" cudss(
      "factorization",
      scheme.linear_problem.solver,
      scheme.linear_problem.x,
      scheme.linear_problem.b,
    )
    @timeit "solve (cold)" cudss(
      "solve",
      scheme.linear_problem.solver,
      scheme.linear_problem.x,
      scheme.linear_problem.b,
    )

    warmup!(scheme)
  else
    @timeit "factorize (warm)" cudss(
      "factorization",
      scheme.linear_problem.solver,
      scheme.linear_problem.x,
      scheme.linear_problem.b,
    )
    @timeit "solve (warm)" cudss(
      "solve",
      scheme.linear_problem.solver,
      scheme.linear_problem.x,
      scheme.linear_problem.b,
    )
  end

  @views begin
    copyto!(u[scheme.halo_aware_indices], scheme.linear_problem.x)
  end

  return nothing
end

function _gpu_iterative_solve!(
  scheme::ImplicitScheme,
  mesh::CurvilinearGrids.AbstractCurvilinearGrid,
  u,
  Δt;
  maxiter=500,
  show_hist=true,
)
  domain_LI = LinearIndices(scheme.domain_indices)

  @timeit "assemble_rhs!" assemble_rhs!(scheme, u, Δt)
  @timeit "assemble_matrix!" assemble_matrix!(scheme, scheme.linear_problem.A, mesh, Δt)
  KernelAbstractions.synchronize(scheme.backend)
  # @timeit "linear solve (warm)" LinearSolve.solve!(scheme.linear_problem)

  if !warmedup(scheme)
    @info "Performing the first solve (cold)"
    @timeit "linear solve (cold)" LinearSolve.solve!(scheme.linear_problem)
    warmup!(scheme)
  else
    @timeit "linear solve (warm)" LinearSolve.solve!(scheme.linear_problem)
  end

  # # precondition
  # @timeit "ilu0! (preconditioner)" ilu0!(scheme.Pl, scheme.A)

  # n = size(scheme.A, 1)
  # precond_op = LinearOperator(
  #   eltype(scheme.A), n, n, false, false, (y, v) -> ldiv!(y, scheme.Pl, v)
  # )

  # if !warmedup(scheme)
  # @timeit "dqgmres! (linear solve)" dqgmres!(
  #   scheme.solver, scheme.A, scheme.b; N=precond_op, history=true, itmax=maxiter
  # )
  #   warmedup!(scheme)
  # else
  #   @timeit "dqgmres! (linear solve)" dqgmres!(
  #     scheme.solver,
  #     scheme.A,
  #     scheme.b,
  #     scheme.solver.x; # use last step as a starting point
  #     N=precond_op,
  #     itmax=maxiter,
  #     history=true,
  #   )
  # end

  # resids = last(scheme.solver.stats.residuals)
  # niter = scheme.solver.stats.niter

  # update u to the solution
  @views begin
    copyto!(u[scheme.halo_aware_indices], scheme.linear_problem.u[domain_LI])
  end

  residual = last(scheme.linear_problem.cacheval.stats.residuals)
  niter = scheme.linear_problem.cacheval.stats.niter
  if show_hist
    @printf "\tConvergence L₂norm : %.1e, solver iterations: %i\n" residual niter
  end

  return nothing
end

@inline function preconditioner(A, ::CPU, τ=0.1)
  return ILUZero.ilu0(A)
end

@inline function preconditioner(A, backend::GPU, τ=0.1)
  return KrylovPreconditioners.kp_ilu0(A)
  # return KrylovPreconditioners.BlockJacobiPreconditioner(A; device=backend)
end

end
