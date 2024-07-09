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
  # display(solver.α)
  # error("done")

  validate_scalar(solver.H, domain, nhalo, :H; enforce_positivity=true)
  validate_scalar(solver.source_term, domain, nhalo, :source_term; enforce_positivity=false)
  validate_scalar(solver.α, domain, nhalo, :diffusivity; enforce_positivity=true)

  # Pseudo-transient iteration
  while err > tol && iter < max_iter && isfinite(err)
    # @info "Iter: $iter, Err: $err"
    iter += 1

    # Diffusion coefficient
    if iter > 1
      update_conductivity!(solver, mesh, solver.H, ρ, cₚ, κ)
    end
    applybcs!(solver.bcs, mesh, solver.H)

    @timeit "update_iteration_params!" update_iteration_params!(solver, ρ, Vpdτ, dt;)

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
        @show extrema(solver.res)
        @error("Non-finite error detected! ($err)")
      end
    end
  end

  if iter == max_iter
    error(
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

@kernel function flux_kernel_cons!(
  qᵢ₊½::AbstractArray{T,2},
  q′ᵢ₊½::AbstractArray{T,2},
  H,
  α,
  θr_dτ,
  edge_metrics,
  axis,
  I0,
  mean_func::F,
) where {T,F}
  idx = @index(Global, Cartesian)
  idx += I0

  @inbounds begin
    i, j = idx.I

    ᵢ₊₁ = shift(idx, axis, +1)

    αᵢ₊½ = 0.5(α[i, j] + α[i + 1, j])
    αⱼ₊½ = 0.5(α[i, j] + α[i, j + 1])

    Jᵢ₊½ = edge_metrics.i₊½.J[i, j]
    Jⱼ₊½ = edge_metrics.j₊½.J[i, j]

    Jξx_ᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[i, j]
    Jξy_ᵢ₊½ = edge_metrics.i₊½.ξ̂.x₂[i, j]

    Jηx_ⱼ₊½ = edge_metrics.j₊½.η̂.x₁[i, j]
    Jηy_ⱼ₊½ = edge_metrics.j₊½.η̂.x₂[i, j]

    a_Jξ²ᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½ + Jξy_ᵢ₊½) / Jᵢ₊½
    a_Jη²ⱼ₊½ = αⱼ₊½ * (Jηx_ⱼ₊½ + Jηy_ⱼ₊½) / Jⱼ₊½
    # a_Jξ²ᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
    # a_Jη²ⱼ₊½ = αⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½

    # edge diffusivity / iter params
    # αᵢ₊½ = mean_func(α[idx], α[ᵢ₊₁])
    θr_dτ_ᵢ₊½ = (θr_dτ[idx] + θr_dτ[ᵢ₊₁]) / 2

    # ∇Hᵢ₊½ = mᵢ₊½ * (H[ᵢ₊₁] - H[idx])
    if axis == 1
      _qᵢ₊½ = -a_Jξ²ᵢ₊½ * (H[ᵢ₊₁] - H[idx])
    else
      _qᵢ₊½ = -a_Jη²ⱼ₊½ * (H[ᵢ₊₁] - H[idx])
    end

    qᵢ₊½[idx] = (qᵢ₊½[idx] * θr_dτ_ᵢ₊½ + _qᵢ₊½) / (1 + θr_dτ_ᵢ₊½)
    q′ᵢ₊½[idx] = _qᵢ₊½
  end
end

@kernel function flux_kernel_noncons!(
  qᵢ₊½::AbstractArray{T,2},
  q′ᵢ₊½::AbstractArray{T,2},
  H,
  α,
  θr_dτ,
  cell_center_metrics,
  axis,
  I0,
  mean_func::F,
) where {T,F}
  idx = @index(Global, Cartesian)
  idx += I0

  @inbounds begin
    ᵢ₊₁ = shift(idx, axis, +1)

    # edge diffusivity / iter params
    αᵢ₊½ = (α[idx] + α[ᵢ₊₁]) / 2
    θr_dτ_ᵢ₊½ = (θr_dτ[idx] + θr_dτ[ᵢ₊₁]) / 2

    _qᵢ₊½ = -αᵢ₊½ * (H[ᵢ₊₁] - H[idx])

    qᵢ₊½[idx] = (qᵢ₊½[idx] * θr_dτ_ᵢ₊½ + _qᵢ₊½) / (1 + θr_dτ_ᵢ₊½)
    q′ᵢ₊½[idx] = _qᵢ₊½
  end
end

function compute_flux!(solver::PseudoTransientSolver{2,T}, mesh) where {T}

  #

  iaxis = 1
  jaxis = 2
  ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)

  # domain = solver.iterators.domain.cartesian
  ᵢ₊½_idx_offset = first(ᵢ₊½_domain) - oneunit(first(ᵢ₊½_domain))
  ⱼ₊½_idx_offset = first(ⱼ₊½_domain) - oneunit(first(ⱼ₊½_domain))

  flux_kernel_noncons!(solver.backend)(
    solver.qH.x,
    solver.qH_2.x,
    solver.H,
    solver.α,
    solver.θr_dτ,
    mesh.cell_center_metrics,
    iaxis,
    ᵢ₊½_idx_offset,
    solver.mean;
    ndrange=size(ᵢ₊½_domain),
  )

  flux_kernel_noncons!(solver.backend)(
    solver.qH.y,
    solver.qH_2.y,
    solver.H,
    solver.α,
    solver.θr_dτ,
    mesh.cell_center_metrics,
    jaxis,
    ⱼ₊½_idx_offset,
    solver.mean;
    ndrange=size(ⱼ₊½_domain),
  )

  KernelAbstractions.synchronize(solver.backend)

  return nothing
end

function compute_flux_cons!(solver::PseudoTransientSolver{2,T}, mesh) where {T}

  #

  iaxis = 1
  jaxis = 2
  ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)

  # domain = solver.iterators.domain.cartesian
  ᵢ₊½_idx_offset = first(ᵢ₊½_domain) - oneunit(first(ᵢ₊½_domain))
  ⱼ₊½_idx_offset = first(ⱼ₊½_domain) - oneunit(first(ⱼ₊½_domain))

  flux_kernel_cons!(solver.backend)(
    solver.qH.x,
    solver.qH_2.x,
    solver.H,
    solver.α,
    solver.θr_dτ,
    mesh.edge_metrics,
    iaxis,
    ᵢ₊½_idx_offset,
    solver.mean;
    ndrange=size(ᵢ₊½_domain),
  )

  flux_kernel_cons!(solver.backend)(
    solver.qH.y,
    solver.qH_2.y,
    solver.H,
    solver.α,
    solver.θr_dτ,
    mesh.edge_metrics,
    jaxis,
    ⱼ₊½_idx_offset,
    solver.mean;
    ndrange=size(ⱼ₊½_domain),
  )

  KernelAbstractions.synchronize(solver.backend)

  return nothing
end

#
@kernel function _update_kernel!(
  H::AbstractArray{T,2},
  H_prev,
  cell_center_metrics, # applying @Const to a struct array causes problems
  edge_metrics, # applying @Const to a struct array causes problems
  α,
  q,
  dτ_ρ,
  source_term,
  dt,
  I0,
) where {T}
  idx = @index(Global, Cartesian)
  idx += I0

  qᵢ = q.x
  qⱼ = q.y

  @inbounds begin
    ∇q = flux_divergence((qᵢ, qⱼ), H, α, cell_center_metrics, edge_metrics, idx)

    H[idx] = (
      (H[idx] + dτ_ρ[idx] * (H_prev[idx] / dt - ∇q + source_term[idx])) /
      (1 + dτ_ρ[idx] / dt)
    )
  end
end

"""
"""
function compute_update!(solver, mesh, Δt)
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _update_kernel!(solver.backend)(
    solver.H,
    solver.H_prev,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    solver.α,
    solver.qH,
    solver.dτ_ρ,
    solver.source_term,
    Δt,
    idx_offset;
    ndrange=size(domain),
  )

  KernelAbstractions.synchronize(solver.backend)
  return nothing
end

"""
    flux_divergence(qH, mesh, idx)

Compute the divergence of the flux, e.g. ∇⋅(α∇H), where the flux is `qH = α∇H`
"""
function flux_divergence_cons(
  (qᵢ, qⱼ), H, α, cell_center_metrics, edge_metrics, idx::CartesianIndex{2}
)
  i, j = idx.I

  # αᵢ₊½ = 0.5(α[i, j] + α[i + 1, j])
  # αᵢ₋½ = 0.5(α[i, j] + α[i - 1, j])
  # αⱼ₊½ = 0.5(α[i, j] + α[i, j + 1])
  # αⱼ₋½ = 0.5(α[i, j] + α[i, j - 1])

  # Jᵢ₊½ = edge_metrics.i₊½.J[i, j]
  # Jⱼ₊½ = edge_metrics.j₊½.J[i, j]
  # Jᵢ₋½ = edge_metrics.i₊½.J[i - 1, j]
  # Jⱼ₋½ = edge_metrics.j₊½.J[i, j - 1]

  # Jξx_ᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[i, j]
  # Jξy_ᵢ₊½ = edge_metrics.i₊½.ξ̂.x₂[i, j]
  # Jηx_ᵢ₊½ = edge_metrics.i₊½.η̂.x₁[i, j]
  # Jηy_ᵢ₊½ = edge_metrics.i₊½.η̂.x₂[i, j]

  # Jξx_ᵢ₋½ = edge_metrics.i₊½.ξ̂.x₁[i - 1, j]
  # Jξy_ᵢ₋½ = edge_metrics.i₊½.ξ̂.x₂[i - 1, j]
  # Jηx_ᵢ₋½ = edge_metrics.i₊½.η̂.x₁[i - 1, j]
  # Jηy_ᵢ₋½ = edge_metrics.i₊½.η̂.x₂[i - 1, j]

  # Jξx_ⱼ₊½ = edge_metrics.j₊½.ξ̂.x₁[i, j]
  # Jξy_ⱼ₊½ = edge_metrics.j₊½.ξ̂.x₂[i, j]
  # Jηx_ⱼ₊½ = edge_metrics.j₊½.η̂.x₁[i, j]
  # Jηy_ⱼ₊½ = edge_metrics.j₊½.η̂.x₂[i, j]

  # Jξx_ⱼ₋½ = edge_metrics.j₊½.ξ̂.x₁[i, j - 1]
  # Jξy_ⱼ₋½ = edge_metrics.j₊½.ξ̂.x₂[i, j - 1]
  # Jηx_ⱼ₋½ = edge_metrics.j₊½.η̂.x₁[i, j - 1]
  # Jηy_ⱼ₋½ = edge_metrics.j₊½.η̂.x₂[i, j - 1]

  # a_Jξ²ᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
  # a_Jξ²ᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2) / Jᵢ₋½
  # a_Jη²ⱼ₊½ = αⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½
  # a_Jη²ⱼ₋½ = αⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2) / Jⱼ₋½

  # a_Jξηᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½ * Jηx_ᵢ₊½ + Jξy_ᵢ₊½ * Jηy_ᵢ₊½) / (4Jᵢ₊½)
  # a_Jξηᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½ * Jηx_ᵢ₋½ + Jξy_ᵢ₋½ * Jηy_ᵢ₋½) / (4Jᵢ₋½)
  # a_Jηξⱼ₊½ = αⱼ₊½ * (Jξx_ⱼ₊½ * Jηx_ⱼ₊½ + Jξy_ⱼ₊½ * Jηy_ⱼ₊½) / (4Jⱼ₊½)
  # a_Jηξⱼ₋½ = αⱼ₋½ * (Jξx_ⱼ₋½ * Jηx_ⱼ₋½ + Jξy_ⱼ₋½ * Jηy_ⱼ₋½) / (4Jⱼ₋½)

  # flux divergence

  # ∇q = (
  #   (a_Jξ²ᵢ₊½ * (H[i + 1, j] - H[i, j]) - a_Jξ²ᵢ₋½ * (H[i, j] - H[i - 1, j])) +
  #   (a_Jη²ⱼ₊½ * (H[i, j + 1] - H[i, j]) - a_Jη²ⱼ₋½ * (H[i, j] - H[i, j - 1]))
  #   # +
  #   # a_Jξηᵢ₊½ * (H[i, j + 1] - H[i, j - 1] + H[i + 1, j + 1] - H[i + 1, j - 1]) -
  #   # a_Jξηᵢ₋½ * (H[i, j + 1] - H[i, j - 1] + H[i - 1, j + 1] - H[i - 1, j - 1]) +
  #   # a_Jηξⱼ₊½ * (H[i + 1, j] - H[i - 1, j] + H[i + 1, j + 1] - H[i - 1, j + 1]) -
  #   # a_Jηξⱼ₋½ * (H[i + 1, j] - H[i - 1, j] + H[i + 1, j - 1] - H[i - 1, j - 1])
  # )
  ∇q = (
    (qᵢ[i, j] - qᵢ[i - 1, j]) + # 
    (qⱼ[i, j] - qⱼ[i, j - 1])
    # + # 
    # a_Jξηᵢ₊½ * (H[i, j + 1] - H[i, j - 1] + H[i + 1, j + 1] - H[i + 1, j - 1]) -
    # a_Jξηᵢ₋½ * (H[i, j + 1] - H[i, j - 1] + H[i - 1, j + 1] - H[i - 1, j - 1]) +
    # a_Jηξⱼ₊½ * (H[i + 1, j] - H[i - 1, j] + H[i + 1, j + 1] - H[i - 1, j + 1]) -
    # a_Jηξⱼ₋½ * (H[i + 1, j] - H[i - 1, j] + H[i + 1, j - 1] - H[i - 1, j - 1])
  )

  # ∇q = ∂qᵢ∂ξ + ∂qⱼ∂η + ∂qᵢ∂η + ∂qⱼ∂ξ + ∂H∂ξ + ∂H∂η
  return ∇q
end

# non-conservative form
function flux_divergence(
  (qᵢ, qⱼ), H, α, cell_center_metrics, edge_metrics, idx::CartesianIndex{2}
)
  i, j = idx.I

  Jᵢ₊½ = edge_metrics.i₊½.J[i, j]
  Jⱼ₊½ = edge_metrics.j₊½.J[i, j]
  Jᵢ₋½ = edge_metrics.i₊½.J[i - 1, j]
  Jⱼ₋½ = edge_metrics.j₊½.J[i, j - 1]

  ξx = cell_center_metrics.ξ.x₁[i, j]
  ξy = cell_center_metrics.ξ.x₂[i, j]
  ηx = cell_center_metrics.η.x₁[i, j]
  ηy = cell_center_metrics.η.x₂[i, j]

  ξxᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[i, j] / Jᵢ₊½
  ξyᵢ₊½ = edge_metrics.i₊½.ξ̂.x₂[i, j] / Jᵢ₊½
  ηxᵢ₊½ = edge_metrics.i₊½.η̂.x₁[i, j] / Jᵢ₊½
  ηyᵢ₊½ = edge_metrics.i₊½.η̂.x₂[i, j] / Jᵢ₊½

  ξxᵢ₋½ = edge_metrics.i₊½.ξ̂.x₁[i - 1, j] / Jᵢ₋½
  ξyᵢ₋½ = edge_metrics.i₊½.ξ̂.x₂[i - 1, j] / Jᵢ₋½
  ηxᵢ₋½ = edge_metrics.i₊½.η̂.x₁[i - 1, j] / Jᵢ₋½
  ηyᵢ₋½ = edge_metrics.i₊½.η̂.x₂[i - 1, j] / Jᵢ₋½

  ξxⱼ₊½ = edge_metrics.j₊½.ξ̂.x₁[i, j] / Jⱼ₊½
  ξyⱼ₊½ = edge_metrics.j₊½.ξ̂.x₂[i, j] / Jⱼ₊½
  ηxⱼ₊½ = edge_metrics.j₊½.η̂.x₁[i, j] / Jⱼ₊½
  ηyⱼ₊½ = edge_metrics.j₊½.η̂.x₂[i, j] / Jⱼ₊½

  ξxⱼ₋½ = edge_metrics.j₊½.ξ̂.x₁[i, j - 1] / Jⱼ₋½
  ξyⱼ₋½ = edge_metrics.j₊½.ξ̂.x₂[i, j - 1] / Jⱼ₋½
  ηxⱼ₋½ = edge_metrics.j₊½.η̂.x₁[i, j - 1] / Jⱼ₋½
  ηyⱼ₋½ = edge_metrics.j₊½.η̂.x₂[i, j - 1] / Jⱼ₋½

  # flux divergence

  aᵢⱼ = (
    ξx * (ξxᵢ₊½ - ξxᵢ₋½) +
    ξy * (ξyᵢ₊½ - ξyᵢ₋½) +
    ηx * (ξxⱼ₊½ - ξxⱼ₋½) +
    ηy * (ξyⱼ₊½ - ξyⱼ₋½)
  )

  bᵢⱼ = (
    ξx * (ηxᵢ₊½ - ηxᵢ₋½) +
    ξy * (ηyᵢ₊½ - ηyᵢ₋½) +
    ηx * (ηxⱼ₊½ - ηxⱼ₋½) +
    ηy * (ηyⱼ₊½ - ηyⱼ₋½)
  )

  ∂qᵢ∂ξ = (ξx^2 + ξy^2) * (qᵢ[i, j] - qᵢ[i - 1, j])
  ∂qⱼ∂η = (ηx^2 + ηy^2) * (qⱼ[i, j] - qⱼ[i, j - 1])

  # ∂qᵢ∂η =
  #   0.25(ηx * ξx + ηy * ξy) * (
  #     α[i + 1, j] * (H[i + 1, j + 1] - H[i + 1, j - 1]) -
  #     α[i - 1, j] * (H[i - 1, j + 1] - H[i - 1, j - 1])
  #   )

  # ∂qⱼ∂ξ =
  #   0.25(ηx * ξx + ηy * ξy) * (
  #     α[i, j + 1] * (H[i + 1, j + 1] - H[i - 1, j + 1]) -
  #     α[i, j - 1] * (H[i + 1, j - 1] - H[i - 1, j - 1])
  #   )

  ∂qᵢ∂η =
    0.25(ηx * ξx + ηy * ξy) * (
      (qᵢ[i, j + 1] + qᵢ[i - 1, j + 1]) - # take average on either side
      (qᵢ[i, j - 1] + qᵢ[i - 1, j - 1])   # and do diff in j
    )

  ∂qⱼ∂ξ =
    0.25(ηx * ξx + ηy * ξy) * (
      (qⱼ[i + 1, j] + qⱼ[i + 1, j - 1]) - # take average on either side
      (qⱼ[i - 1, j] + qⱼ[i - 1, j - 1])   # and do diff in i
    )

  ∂H∂ξ = aᵢⱼ * 0.5(qᵢ[i, j] + qᵢ[i - 1, j]) # ∂u/∂ξ + non-orth terms
  ∂H∂η = bᵢⱼ * 0.5(qⱼ[i, j] + qⱼ[i, j - 1]) # ∂u/∂η + non-orth terms
  # ∂H∂ξ = 0.5α[i, j] * aᵢⱼ * (H[i + 1, j] - H[i - 1, j])  # ∂H/∂ξ + non-orth terms
  # ∂H∂η = 0.5α[i, j] * bᵢⱼ * (H[i, j + 1] - H[i, j - 1])  # ∂H/∂η + non-orth terms

  # ∂qᵢ∂ξ = ∂qᵢ∂ξ * (abs(∂qᵢ∂ξ) <= 1e-14)
  # ∂qⱼ∂η = ∂qⱼ∂η * (abs(∂qⱼ∂η) <= 1e-14)
  # ∂qᵢ∂η = ∂qᵢ∂η * (abs(∂qᵢ∂η) <= 1e-14)
  # ∂qⱼ∂ξ = ∂qⱼ∂ξ * (abs(∂qⱼ∂ξ) <= 1e-14)
  # ∂H∂ξ = ∂H∂ξ * (abs(∂H∂ξ) <= 1e-14)
  # ∂H∂η = ∂H∂η * (abs(∂H∂η) <= 1e-14)

  ∇q = ∂qᵢ∂ξ + ∂qⱼ∂η + ∂qᵢ∂η + ∂qⱼ∂ξ + ∂H∂ξ + ∂H∂η
  return ∇q
end

function flux_divergence_orig(
  (qHx, qHy), H, α, cell_center_metrics, edge_metrics, idx::CartesianIndex{2}
)
  m = non_conservative_metrics(cell_center_metrics, edge_metrics, idx)

  iaxis = 1
  jaxis = 2
  ᵢ₋₁ = shift(idx, iaxis, -1)
  ⱼ₋₁ = shift(idx, jaxis, -1)

  αᵢⱼ = (
    m.ξx * (m.ξxᵢ₊½ - m.ξxᵢ₋½) +
    m.ξy * (m.ξyᵢ₊½ - m.ξyᵢ₋½) +
    m.ηx * (m.ξxⱼ₊½ - m.ξxⱼ₋½) +
    m.ηy * (m.ξyⱼ₊½ - m.ξyⱼ₋½)
  )

  βᵢⱼ = (
    m.ξx * (m.ηxᵢ₊½ - m.ηxᵢ₋½) +
    m.ξy * (m.ηyᵢ₊½ - m.ηyᵢ₋½) +
    m.ηx * (m.ηxⱼ₊½ - m.ηxⱼ₋½) +
    m.ηy * (m.ηyⱼ₊½ - m.ηyⱼ₋½)
  )

  ∇qH = (
    # (qHx[idx] - qHx[ᵢ₋₁]) + # ∂qH∂x
    # (qHy[idx] - qHy[ⱼ₋₁])   # ∂qH∂y
    (m.ξx^2 + m.ξy^2) * (qHx[idx] - qHx[ᵢ₋₁]) + # ∂qH∂x
    (m.ηx^2 + m.ηy^2) * (qHy[idx] - qHy[ⱼ₋₁])   # ∂qH∂y
  )

  return ∇qH
end

@kernel function _update_resid!(
  resid::AbstractArray{T,2},
  cell_center_metrics,
  edge_metrics,
  α,
  H,
  H_prev,
  flux,
  source_term,
  dt,
  I0,
) where {T}
  idx = @index(Global, Cartesian)
  idx += I0

  qᵢ = flux.x
  qⱼ = flux.y

  ∇q = flux_divergence((qᵢ, qⱼ), H, α, cell_center_metrics, edge_metrics, idx)

  resid[idx] = -(H[idx] - H_prev[idx]) / dt - ∇q + source_term[idx]
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
    mesh.edge_metrics,
    solver.α,
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
