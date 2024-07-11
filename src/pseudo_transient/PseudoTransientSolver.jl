module PseudoTransientScheme

using LinearAlgebra: norm

using TimerOutputs: @timeit
using CartesianDomains: expand, shift, expand_lower, haloedge_regions
using CurvilinearGrids: CurvilinearGrid2D, CurvilinearGrid3D, cellsize_withhalo, coords
using KernelAbstractions
using KernelAbstractions: CPU, GPU, @kernel, @index
using Polyester: @batch
using UnPack: @unpack

using ..BoundaryConditions

include("../averaging.jl")
include("../validity_checks.jl")
include("../edge_terms.jl")

include("conductivity.jl")
include("flux.jl")
include("flux_divergence.jl")
include("residuals.jl")
include("update.jl")

export PseudoTransientSolver

struct PseudoTransientSolver{N,T,BE,AA<:AbstractArray{T,N},NT1,DM,MC,B,F}
  u::AA
  u_prev::AA
  source_term::AA
  q::NT1
  q′::NT1
  residual::AA
  Reynolds_number::AA
  α::AA # diffusivity
  θr_dτ::AA
  dτ_ρ::AA
  spacing::NTuple{N,T}
  L::T
  iterators::DM
  metric_cache::MC
  bcs::B # boundary conditions
  mean::F
  backend::BE
end

function PseudoTransientSolver(
  mesh, bcs; backend=CPU(), face_diffusivity=:arithmetic, T=Float64
)
  #
  #         u
  #     |--------|--------|--------|--------|
  #           q_i+1/2
  #

  iterators = (
    domain=(cartesian=mesh.iterators.cell.domain,),
    full=(cartesian=mesh.iterators.cell.full,),
  )

  # cell-based
  u = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  u_prev = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  S = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)) # source term
  residual = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  Reynolds_number = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  α = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  θr_dτ = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  dτ_ρ = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))

  # edge-based; same dims as the cell-based arrays, but each entry stores the {i,j,k}+1/2 value
  q = flux_tuple(mesh)
  q′ = flux_tuple(mesh)

  L, spacing = phys_dims(mesh)
  if face_diffusivity === :harmonic
    mean_func = harmonic_mean # from ../averaging.jl
  else
    mean_func = arithmetic_mean # from ../averaging.jl
  end

  metric_cache = nothing

  return PseudoTransientSolver(
    u,
    u_prev,
    S,
    q,
    q′,
    residual,
    Reynolds_number,
    α,
    θr_dτ,
    dτ_ρ,
    spacing,
    L,
    iterators,
    metric_cache,
    bcs,
    mean_func,
    backend,
  )
end

function phys_dims(mesh::CurvilinearGrid2D)
  x, y = coords(mesh)
  spacing = (minimum(diff(x; dims=1)), minimum(diff(y; dims=2))) .|> T
  min_x, max_x = extrema(x)
  min_y, max_y = extrema(y)
  L = max(abs(max_x - min_x), abs(max_y - min_y)) |> T
  # L = maximum(spacing) 

  return L, spacing
end

function phys_dims(mesh::CurvilinearGrid3D)
  x, y, z = coords(mesh)
  spacing =
    (minimum(diff(x; dims=1)), minimum(diff(y; dims=2)), minimum(diff(z; dims=3))) .|> T

  min_x, max_x = extrema(x)
  min_y, max_y = extrema(y)
  min_z, max_z = extrema(z)

  L = max(abs(max_x - min_x), abs(max_y - min_y), abs(max_z - min_z)) |> T
  # L = maximum(spacing)
  return L, spacing
end

function flux_tuple(mesh::CurvilinearGrid2D)
  return (
    x=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    y=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
  )
end

function flux_tuple(mesh::CurvilinearGrid3D)
  return (
    x=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    y=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    z=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
  )
end

# ------------------------------------------------------------------------------------------------

# solve a single time-step dt
function step!(
  solver::PseudoTransientSolver{N},
  mesh,
  T,
  ρ,
  cₚ,
  κ,
  dt;
  max_iter=1e5,
  tol=1e-8,
  error_check_interval=10,
  cutoff=true,
) where {N}

  #
  domain = solver.iterators.domain.cartesian
  nhalo = 1

  iter = 0
  err = 2 * tol

  CFL = (1 / sqrt(N))

  dx, dy = solver.spacing
  Vpdτ = CFL * min(dx, dy)

  copy!(solver.u, T)
  copy!(solver.u_prev, solver.u)

  update_conductivity!(solver, mesh, solver.u, ρ, cₚ, κ)

  fill!(solver.α, 1.0)
  validate_scalar(solver.u, domain, nhalo, :u; enforce_positivity=true)
  validate_scalar(solver.source_term, domain, nhalo, :source_term; enforce_positivity=false)
  validate_scalar(solver.α, domain, nhalo, :diffusivity; enforce_positivity=true)

  # Pseudo-transient iteration
  while err > tol && iter < max_iter && isfinite(err)
    # @info "Iter: $iter"
    iter += 1

    # Diffusion coefficient
    if iter > 1
      update_conductivity!(solver, mesh, solver.u, ρ, cₚ, κ)
    end

    @timeit "update_iteration_params!" update_iteration_params!(solver, ρ, Vpdτ, dt;)

    applybcs!(solver.bcs, mesh, solver.u)
    applybcs!(solver.bcs, mesh, solver.α)
    applybcs!(solver.bcs, mesh, solver.dτ_ρ)
    applybcs!(solver.bcs, mesh, solver.θr_dτ)

    @timeit "compute_flux!" compute_flux!(solver, mesh)
    @timeit "compute_update!" compute_update!(solver, mesh, dt)

    # Apply a cutoff function to remove negative
    if cutoff
      cutoff!(solver.u)
    end

    if iter % error_check_interval == 0 || iter == 1
      @timeit "update_residual!" update_residual!(solver, mesh, dt)
      @timeit "norm" begin
        inner_dom = solver.iterators.domain.cartesian
        residual = @view solver.residual[inner_dom]
        err2 = L2_norm(residual)
        derr = abs(err - err2)
        if derr < 1
          break
        end
        # @info "\tErr: $err2, $derr"
        err = err2
      end
      if !isfinite(err)
        @show extrema(solver.residual)
        error("Non-finite error detected! ($err)")
      end
    end
  end

  if iter == max_iter
    @error(
      "Maximum iteration limit reached ($max_iter), current error is $(err), tolerance is $tol, exiting...",
    )
  end

  validate_scalar(solver.u, domain, nhalo, :u; enforce_positivity=true)
  copy!(solver.u_prev, solver.u)
  copy!(T, solver.u)

  return err, iter
end

# ------------------------------------------------------------------------------------------------

function L2_norm(A)
  _norm = sqrt(mapreduce(x -> x^2, +, A)) / sqrt(length(A))
  return _norm
end

function update_iteration_params!(solver, ρ, Vpdτ, Δt; iter_scale=1)
  @kernel function _iter_param_kernel!(
    Reynolds_number, dτ_ρ, θr_dτ, _Vpdτ, L, _ρ, α, dt, β, index_offset
  )
    idx = @index(Global, Cartesian)
    idx += index_offset

    @inbounds begin
      _Re = π + sqrt(π^2 + (L^2 * _ρ[idx]) / (α[idx] * dt))
      Reynolds_number[idx] = _Re
      dτ_ρ[idx] = (_Vpdτ * L / (α[idx] * _Re)) * β
      θr_dτ[idx] = (L / (_Vpdτ * _Re)) * β
    end
  end

  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _iter_param_kernel!(solver.backend)(
    solver.Reynolds_number,
    solver.dτ_ρ,
    solver.θr_dτ,
    Vpdτ,
    solver.L,
    ρ,
    solver.α,
    Δt,
    iter_scale,
    idx_offset;
    ndrange=size(domain),
  )

  KernelAbstractions.synchronize(solver.backend)
  return nothing
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

end # module
