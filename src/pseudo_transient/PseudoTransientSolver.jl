module PseudoTransientScheme

using LinearAlgebra: norm

using TimerOutputs: @timeit
using CartesianDomains: expand, shift, expand_lower, haloedge_regions
using CurvilinearGrids: CurvilinearGrid2D, cellsize_withhalo, coords
using KernelAbstractions
using KernelAbstractions: CPU, GPU, @kernel, @index
using Polyester: @batch
using UnPack: @unpack

using ..BoundaryConditions

include("../averaging.jl")
include("../validity_checks.jl")
include("../edge_terms.jl")

export PseudoTransientSolver

struct PseudoTransientSolver{N,T,BE,AA<:AbstractArray{T,N},NT1,DM,B,F}
  H::AA
  H_prev::AA
  source_term::AA
  qH::NT1
  qH_2::NT1
  res::AA
  Re::AA
  α::AA # diffusivity
  θr_dτ::AA
  dτ_ρ::AA
  spacing::NTuple{N,T}
  L::T
  iterators::DM
  bcs::B # boundary conditions
  mean::F
  backend::BE
end

function PseudoTransientSolver(
  mesh::CurvilinearGrid2D, bcs; backend=CPU(), face_diffusivity=:arithmetic, T=Float64
)
  #
  #         H
  #     |--------|--------|--------|--------|
  #           q_i+1/2
  #

  iterators = (
    domain=(cartesian=mesh.iterators.cell.domain,),
    full=(cartesian=mesh.iterators.cell.full,),
  )

  # cell-based
  H = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  H_prev = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  S = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)) # source term

  # edge-based
  qH = (
    x=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    y=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
  )
  qH² = (
    x=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    y=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
  )

  res = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  Re = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))

  α = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))

  θr_dτ = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  dτ_ρ = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))

  x, y = coords(mesh)
  spacing = (minimum(diff(x; dims=1)), minimum(diff(y; dims=2))) .|> T
  min_x, max_x = extrema(x)
  min_y, max_y = extrema(y)
  L = max(abs(max_x - min_x), abs(max_y - min_y)) |> T

  if face_diffusivity === :harmonic
    mean_func = harmonic_mean # from ../averaging.jl
  else
    mean_func = arithmetic_mean # from ../averaging.jl
  end

  return PseudoTransientSolver(
    H,
    H_prev,
    S,
    qH,
    qH²,
    res,
    Re,
    α,
    θr_dτ,
    dτ_ρ,
    spacing,
    L,
    iterators,
    bcs,
    mean_func,
    backend,
  )
end

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

  CFL = 1 / sqrt(N)

  dx, dy = solver.spacing
  Vpdτ = CFL * min(dx, dy)

  copy!(solver.H, T)
  copy!(solver.H_prev, solver.H)

  update_conductivity!(solver, mesh, solver.H, ρ, cₚ, κ)

  validate_scalar(solver.H, domain, nhalo, :H; enforce_positivity=true)
  validate_scalar(solver.source_term, domain, nhalo, :source_term; enforce_positivity=false)
  validate_scalar(solver.α, domain, nhalo, :diffusivity; enforce_positivity=true)

  # Pseudo-transient iteration
  while err > tol && iter < max_iter && isfinite(err)
    iter += 1

    # Diffusion coefficient
    if iter > 1
      update_conductivity!(solver, mesh, solver.H, ρ, cₚ, κ)
    end
    applybcs!(solver.bcs, mesh, solver.H)

    @timeit "update_iteration_params!" update_iteration_params!(solver, ρ, Vpdτ, dt)

    applybcs!(solver.bcs, mesh, solver.α)
    applybcs!(solver.bcs, mesh, solver.dτ_ρ)
    applybcs!(solver.bcs, mesh, solver.θr_dτ)

    @timeit "compute_flux!" compute_flux!(solver, mesh)
    @timeit "compute_update!" compute_update!(solver, mesh, dt)

    # Apply a cutoff function to remove negative
    # if cutoff
    cutoff!(solver.H)
    # end

    if iter % error_check_interval == 0
      @timeit "update_residual!" update_residual!(solver, mesh, dt)

      @timeit "norm" begin
        res = @view solver.res[solver.iterators.domain.cartesian]

        err = L2_norm(res)

        # err = norm(res) / sqrt(length(res))
      end
      if !isfinite(err)
        error("Non-finite error detected! ($err)")
      end
    end
  end

  if iter == max_iter
    @error(
      "Maximum iteration limit reached ($max_iter), current error is $(err), tolerance is $tol, exiting...",
    )
  end

  validate_scalar(solver.H, domain, nhalo, :H; enforce_positivity=true)
  copy!(solver.H_prev, solver.H)
  copy!(T, solver.H)

  return err, iter
end

function L2_norm(A)
  _norm = sqrt(mapreduce(x -> x^2, +, A)) / sqrt(length(A))
  return _norm
end

function update_iteration_params!(solver, ρ, Vpdτ, Δt; iter_scale=1)
  # β=1 / 1.2, # iteration scaling parameter
  # β = 1, # iteration scaling parameter
  #
  @kernel function _iter_param_kernel!(Re, dτ_ρ, θr_dτ, _Vpdτ, L, _ρ, α, dt, β, I0)
    idx = @index(Global, Cartesian)
    idx += I0

    @inbounds begin
      _Re = π + sqrt(π^2 + (L^2 * _ρ[idx]) / (α[idx] * dt))
      Re[idx] = _Re
      dτ_ρ[idx] = (_Vpdτ * L / (α[idx] * _Re)) * β
      θr_dτ[idx] = (L / (_Vpdτ * _Re)) * β
    end
  end

  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _iter_param_kernel!(solver.backend)(
    solver.Re,
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

  # for idx in solver.iterators.domain.cartesian
  #   solver.Re[idx] = π + sqrt(π^2 + (solver.L^2 * ρ[idx]) / (solver.α[idx] * dt))
  # end

  # for idx in solver.iterators.domain.cartesian
  #   solver.dτ_ρ[idx] = (Vpdτ * solver.L / (solver.α[idx] * solver.Re[idx])) * β
  #   solver.θr_dτ[idx] = (solver.L / (Vpdτ * solver.Re[idx])) * β
  # end

  # return nothing
end

@inline function metric(m, axis, ::CartesianIndex{2})
  if axis == 1
    return m.ξx + m.ξy
  else
    return m.ηx + m.ηy
  end
end

@kernel function flux_kernel!(
  qHxᵢ₊½,
  qHx_2ᵢ₊½,
  @Const(H),
  @Const(α),
  @Const(θr_dτ),
  cell_center_metrics,
  axis,
  I0,
  # mean_func::F,
)
  idx = @index(Global, Cartesian)
  idx += I0

  harmon_mean(a, b) = (2a * b) / (a + b)

  @inbounds begin
    # m = non_conservative_metrics_iso(cell_center_metrics, edge_metrics, idx)

    if axis == 1
      ξx = cell_center_metrics.ξ.x₁[idx]
      ξy = cell_center_metrics.ξ.x₂[idx]
      mᵢ₊½ = ξx + ξy
    else
      ηx = cell_center_metrics.η.x₁[idx]
      ηy = cell_center_metrics.η.x₂[idx]
      mᵢ₊½ = ηx + ηy
    end

    # mᵢ₊½ = metric(m, axis, idx)

    ᵢ₊₁ = shift(idx, axis, +1)

    # edge diffusivity / iter params
    αᵢ₊½ = harmon_mean(α[idx], α[ᵢ₊₁])
    θr_dτ_ᵢ₊½ = (θr_dτ[idx] + θr_dτ[ᵢ₊₁]) / 2

    ∂H∂xᵢ₊½ = mᵢ₊½ * (H[ᵢ₊₁] - H[idx])

    qHxᵢ₊½[idx] = (qHxᵢ₊½[idx] * θr_dτ_ᵢ₊½ - αᵢ₊½ * ∂H∂xᵢ₊½) / (1 + θr_dτ_ᵢ₊½)
    qHx_2ᵢ₊½[idx] = -αᵢ₊½ * ∂H∂xᵢ₊½
  end
end

@inline harm_mean(a, b) = (2a * b) / (a + b)

function compute_flux!(solver::PseudoTransientSolver{2,T}, mesh) where {T}

  #

  iaxis = 1
  jaxis = 2
  ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)

  # domain = solver.iterators.domain.cartesian
  ᵢ₊½_idx_offset = first(ᵢ₊½_domain) - oneunit(first(ᵢ₊½_domain))
  ⱼ₊½_idx_offset = first(ⱼ₊½_domain) - oneunit(first(ⱼ₊½_domain))

  flux_kernel!(solver.backend)(
    solver.qH.x,
    solver.qH_2.x,
    solver.H,
    solver.α,
    solver.θr_dτ,
    mesh.cell_center_metrics,
    iaxis,
    ᵢ₊½_idx_offset;
    # solver.mean;
    ndrange=size(ᵢ₊½_domain),
  )

  flux_kernel!(solver.backend)(
    solver.qH.y,
    solver.qH_2.y,
    solver.H,
    solver.α,
    solver.θr_dτ,
    mesh.cell_center_metrics,
    jaxis,
    ⱼ₊½_idx_offset;
    # solver.mean;
    ndrange=size(ⱼ₊½_domain),
  )

  KernelAbstractions.synchronize(solver.backend)

  # qHxᵢ₊½ = solver.qH.x
  # qHyⱼ₊½ = solver.qH.y

  # θr_dτ = solver.θr_dτ
  # qHx_2ᵢ₊½ = solver.qH_2.x
  # qHy_2ⱼ₊½ = solver.qH_2.y

  # # fluxes are on the edge, H_i+1/2
  # iaxis, jaxis = (1, 2)

  # dx, dy = solver.spacing

  # for idx in ᵢ₊½_domain
  #   m = non_conservative_metrics_iso(mesh.cell_center_metrics, mesh.edge_metrics, idx)
  #   ᵢ₊₁ = shift(idx, iaxis, +1)

  #   # edge diffusivity / iter params
  #   αᵢ₊½ = solver.mean(solver.α[idx], solver.α[ᵢ₊₁])
  #   θr_dτ_ᵢ₊½ = arithmetic_mean(θr_dτ[idx], θr_dτ[ᵢ₊₁])

  #   # ∂H∂xᵢ₊½ = (solver.H[ᵢ₊₁] - solver.H[idx]) / dx
  #   ∂H∂xᵢ₊½ = (m.ξx + m.ξy) * (solver.H[ᵢ₊₁] - solver.H[idx])

  #   qHxᵢ₊½[idx] = (qHxᵢ₊½[idx] * θr_dτ_ᵢ₊½ - αᵢ₊½ * ∂H∂xᵢ₊½) / (1 + θr_dτ_ᵢ₊½)

  #   qHx_2ᵢ₊½[idx] = -αᵢ₊½ * ∂H∂xᵢ₊½
  # end

  # for idx in ⱼ₊½_domain
  #   m = non_conservative_metrics_iso(mesh.cell_center_metrics, mesh.edge_metrics, idx)
  #   ⱼ₊₁ = shift(idx, jaxis, +1)

  #   # edge diffusivity / iter params
  #   αⱼ₊½ = solver.mean(solver.α[idx], solver.α[ⱼ₊₁])
  #   θr_dτ_ⱼ₊½ = arithmetic_mean(θr_dτ[idx], θr_dτ[ⱼ₊₁])

  #   # ∂H∂yⱼ₊½ = (solver.H[ⱼ₊₁] - solver.H[idx]) / dy
  #   ∂H∂yⱼ₊½ = (m.ηx + m.ηy) * (solver.H[ⱼ₊₁] - solver.H[idx])

  #   qHyⱼ₊½[idx] = (qHyⱼ₊½[idx] * θr_dτ_ⱼ₊½ - αⱼ₊½ * ∂H∂yⱼ₊½) / (1 + θr_dτ_ⱼ₊½)

  #   qHy_2ⱼ₊½[idx] = -αⱼ₊½ * ∂H∂yⱼ₊½
  # end

  return nothing
end

#
@kernel function _update_kernel!(
  H,
  @Const(H_prev),
  cell_center_metrics,
  @Const(qH),
  @Const(dτ_ρ),
  @Const(source_term),
  dt,
  I0,
)
  idx = @index(Global, Cartesian)
  idx += I0

  @inbounds begin
    # ∇qH = flux_divergence(qH, grid, idx)

    qHx = qH.x
    qHy = qH.y

    # m = non_conservative_metrics(mesh.cell_center_metrics, mesh.edge_metrics, idx)

    ξx = cell_center_metrics.ξ.x₁[idx]
    ξy = cell_center_metrics.ξ.x₂[idx]
    ηx = cell_center_metrics.η.x₁[idx]
    ηy = cell_center_metrics.η.x₂[idx]

    iaxis = 1
    jaxis = 2
    ᵢ₋₁ = shift(idx, iaxis, -1)
    ⱼ₋₁ = shift(idx, jaxis, -1)

    ∇qH = (
      (ξx + ξy) * (qHx[idx] - qHx[ᵢ₋₁]) + # ∂qH∂x
      (ηx + ηy) * (qHy[idx] - qHy[ⱼ₋₁])   # ∂qH∂y
    )

    H[idx] = (
      (H[idx] + dτ_ρ[idx] * (H_prev[idx] / dt - ∇qH + source_term[idx])) /
      (1 + dτ_ρ[idx] / dt)
    )
  end
end

"""
"""
function compute_update!(solver, mesh, Δt)
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  workgroup = (32, 32)

  _update_kernel!(solver.backend)(
    solver.H,
    solver.H_prev,
    mesh.cell_center_metrics,
    solver.qH,
    solver.dτ_ρ,
    solver.source_term,
    Δt,
    idx_offset;
    ndrange=size(domain),
  )

  KernelAbstractions.synchronize(solver.backend)
  return nothing

  # for idx in solver.iterators.domain.cartesian
  #   # ᵢ₋₁ = shift(idx, i, -1)
  #   # ⱼ₋₁ = shift(idx, j, -1)

  #   # m = non_conservative_metrics_iso(mesh.cell_center_metrics, mesh.edge_metrics, idx)

  #   # ∇qH = (
  #   #   (m.ξx^2 + m.ξy^2) * (qHx[idx] - qHx[ᵢ₋₁]) + # ∂qH∂x
  #   #   (m.ηx^2 + m.ηy^2) * (qHy[idx] - qHy[ⱼ₋₁])   # ∂qH∂y
  #   # ) # + solver.source_term[idx]
  #   # ∇qH = (
  #   #   (qHx[idx] - qHx[ᵢ₋₁]) / dx + # ∂qH∂x
  #   #   (qHy[idx] - qHy[ⱼ₋₁]) / dy   # ∂qH∂y
  #   # ) # + solver.source_term[idx]

  #   ∇qH = flux_divergence(solver.qH, mesh, idx)

  #   solver.H[idx] = (
  #     (
  #       solver.H[idx] +
  #       solver.dτ_ρ[idx] * (solver.H_prev[idx] / dt - ∇qH + solver.source_term[idx])
  #     ) / (1 + solver.dτ_ρ[idx] / dt)
  #   )
  # end

end

"""
    flux_divergence(qH, mesh, idx)

Compute the divergence of the flux, e.g. ∇⋅(α∇H), where the flux is `qH = α∇H`
"""
function flux_divergence(qH, mesh, idx::CartesianIndex{2})
  qHx = qH.x
  qHy = qH.y

  m = non_conservative_metrics(mesh.cell_center_metrics, mesh.edge_metrics, idx)

  iaxis = 1
  jaxis = 2
  ᵢ₋₁ = shift(idx, iaxis, -1)
  ⱼ₋₁ = shift(idx, jaxis, -1)

  # αᵢⱼ = (
  #   m.ξx * (m.ξxᵢ₊½ - m.ξxᵢ₋½) +
  #   m.ξy * (m.ξyᵢ₊½ - m.ξyᵢ₋½) +
  #   m.ηx * (m.ξxⱼ₊½ - m.ξxⱼ₋½) +
  #   m.ηy * (m.ξyⱼ₊½ - m.ξyⱼ₋½)
  # )

  # βᵢⱼ = (
  #   m.ξx * (m.ηxᵢ₊½ - m.ηxᵢ₋½) +
  #   m.ξy * (m.ηyᵢ₊½ - m.ηyᵢ₋½) +
  #   m.ηx * (m.ηxⱼ₊½ - m.ηxⱼ₋½) +
  #   m.ηy * (m.ηyⱼ₊½ - m.ηyⱼ₋½)
  # )

  ∇qH = (
    (m.ξx + m.ξy) * (qHx[idx] - qHx[ᵢ₋₁]) + # ∂qH∂x
    (m.ηx + m.ηy) * (qHy[idx] - qHy[ⱼ₋₁])   # ∂qH∂y
  )

  return ∇qH
end

@kernel function _update_resid!(
  resid, cell_center_metrics, H, H_prev, flux, source_term, dt, I0
)
  idx = @index(Global, Cartesian)
  idx += I0

  qHx = flux.x
  qHy = flux.y

  @inbounds begin

    # m = non_conservative_metrics(mesh.cell_center_metrics, mesh.edge_metrics, idx)

    ξx = cell_center_metrics.ξ.x₁[idx]
    ξy = cell_center_metrics.ξ.x₂[idx]
    ηx = cell_center_metrics.η.x₁[idx]
    ηy = cell_center_metrics.η.x₂[idx]

    iaxis = 1
    jaxis = 2
    ᵢ₋₁ = shift(idx, iaxis, -1)
    ⱼ₋₁ = shift(idx, jaxis, -1)

    ∇qH = (
      (ξx + ξy) * (qHx[idx] - qHx[ᵢ₋₁]) + # ∂qH∂x
      (ηx + ηy) * (qHy[idx] - qHy[ⱼ₋₁])   # ∂qH∂y
    )

    # ∇qH = flux_divergence(flux, grid, idx)

    resid[idx] = -(H[idx] - H_prev[idx]) / dt - ∇qH + source_term[idx]
  end
end

"""
    update_residual!(solver::PseudoTransientSolver, mesh, Δt)

"""
function update_residual!(solver::PseudoTransientSolver, mesh, Δt)
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _update_resid!(solver.backend)(
    solver.res,
    mesh.cell_center_metrics,
    solver.H,
    solver.H_prev,
    solver.qH_2,
    solver.source_term,
    Δt,
    idx_offset;
    ndrange=size(domain),
  )

  KernelAbstractions.synchronize(solver.backend)
  return nothing
end

# ------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------
function update_conductivity!(
  scheme, mesh, temperature, density, cₚ::Real, κ::F
) where {F<:Function}
  @unpack diff_domain, domain = _domain_pairs(scheme, mesh)

  α = @view scheme.α[diff_domain]
  T = @view temperature[domain]
  ρ = @view density[domain]

  backend = scheme.backend
  conductivity_kernel(backend)(α, T, ρ, cₚ, κ; ndrange=size(α))

  KernelAbstractions.synchronize(backend)

  return nothing
end

function update_conductivity!(
  scheme, mesh, temperature, density, cₚ::AbstractArray, κ::F
) where {F<:Function}
  @unpack diff_domain, domain = _domain_pairs(scheme, mesh)

  α = @view scheme.α[diff_domain]
  T = @view temperature[domain]
  _cₚ = @view cₚ[domain]
  ρ = @view density[domain]

  backend = scheme.backend
  conductivity_kernel(backend)(α, T, ρ, _cₚ, κ; ndrange=size(α))

  KernelAbstractions.synchronize(backend)

  return nothing
end

function _domain_pairs(scheme::PseudoTransientSolver, mesh)
  diff_domain = scheme.iterators.full.cartesian
  domain = mesh.iterators.cell.full

  return (; diff_domain, domain)
end

# conductivity with array-valued cₚ
@kernel function conductivity_kernel(
  α, temperature, density, cₚ::AbstractArray{T,N}, κ::F
) where {T,N,F<:Function}
  idx = @index(Global)

  @inbounds begin
    α[idx] = κ(density[idx], temperature[idx]) / (density[idx] * cₚ[idx])
  end
end

# conductivity with single-valued cₚ
@kernel function conductivity_kernel(
  α, temperature, density, cₚ::Real, κ::F
) where {F<:Function}
  idx = @index(Global)

  @inbounds begin
    α[idx] = abs(κ(density[idx], temperature[idx]) / (density[idx] * cₚ))
  end
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
