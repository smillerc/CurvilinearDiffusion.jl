module PseudoTransientScheme

using LinearAlgebra: norm

using CartesianDomains: expand, shift, expand_lower, haloedge_regions
using CurvilinearGrids: CurvilinearGrid2D, cellsize_withhalo, coords
using KernelAbstractions
using KernelAbstractions: CPU, GPU, @kernel, @index
using Polyester: @batch
using UnPack: @unpack

using ..BoundaryConditions

include("../averaging.jl")
include("../validity_checks.jl")

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
  spacing = (minimum(diff(x; dims=1)), minimum(diff(y; dims=2)))
  min_x, max_x = extrema(x)
  min_y, max_y = extrema(y)
  L = max(abs(max_x - min_x), abs(max_y - min_y))

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
  # display(solver.α)
  validate_scalar(solver.α, domain, nhalo, :diffusivity; enforce_positivity=true)

  # Pseudo-transient iteration
  while err > tol && iter < max_iter && isfinite(err)
    iter += 1

    # Diffusion coefficient
    if iter > 1
      update_conductivity!(solver, mesh, solver.H, ρ, cₚ, κ)
    end
    applybcs!(solver.bcs, mesh, solver.H)

    update_iteration_params!(solver, ρ, Vpdτ, dt)

    applybcs!(solver.bcs, mesh, solver.α)
    applybcs!(solver.bcs, mesh, solver.dτ_ρ)
    applybcs!(solver.bcs, mesh, solver.θr_dτ)

    compute_flux!(solver)
    compute_update!(solver, dt)

    if iter % error_check_interval == 0
      update_residual!(solver, dt)

      res = @view solver.res[solver.iterators.domain.cartesian]
      err = norm(res) / sqrt(length(res))

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

  # Apply a cutoff function to remove negative u values
  if cutoff
    cutoff!(solver.H)
  end

  validate_scalar(solver.H, domain, nhalo, :H; enforce_positivity=true)
  copy!(solver.H_prev, solver.H)
  copy!(T, solver.H)

  return err, iter
end

function update_iteration_params!(
  solver,
  ρ,
  Vpdτ,
  dt,
  # β=1 / 1.2, # iteration scaling parameter
  β=1, # iteration scaling parameter
)
  @inbounds for idx in solver.iterators.domain.cartesian
    solver.Re[idx] = π + sqrt(π^2 + (solver.L^2 * ρ[idx]) / (solver.α[idx] * dt))
  end

  @inbounds for idx in solver.iterators.domain.cartesian
    solver.dτ_ρ[idx] = (Vpdτ * solver.L / (solver.α[idx] * solver.Re[idx])) * β
    solver.θr_dτ[idx] = (solver.L / (Vpdτ * solver.Re[idx])) * β
  end

  return nothing
end

function compute_flux!(solver::PseudoTransientSolver{2,T}) where {T}
  qHxᵢ₊½ = solver.qH.x
  qHyⱼ₊½ = solver.qH.y

  θr_dτ = solver.θr_dτ
  qHx_2ᵢ₊½ = solver.qH_2.x
  qHy_2ⱼ₊½ = solver.qH_2.y

  # fluxes are on the edge, H_i+1/2
  i, j = (1, 2)

  dx, dy = solver.spacing

  ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, i, +1)
  @inbounds for idx in ᵢ₊½_domain
    ᵢ₊₁ = shift(idx, i, +1)

    # edge diffusivity / iter params
    αᵢ₊½ = solver.mean(solver.α[idx], solver.α[ᵢ₊₁])
    θr_dτ_ᵢ₊½ = arithmetic_mean(θr_dτ[idx], θr_dτ[ᵢ₊₁])

    ∂H∂xᵢ₊½ = (solver.H[ᵢ₊₁] - solver.H[idx]) / dx

    qHxᵢ₊½[idx] = (qHxᵢ₊½[idx] * θr_dτ_ᵢ₊½ - αᵢ₊½ * ∂H∂xᵢ₊½) / (1 + θr_dτ_ᵢ₊½)

    qHx_2ᵢ₊½[idx] = -αᵢ₊½ * ∂H∂xᵢ₊½
  end

  ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, j, +1)
  @inbounds for idx in ⱼ₊½_domain
    ⱼ₊₁ = shift(idx, j, +1)

    # edge diffusivity / iter params
    αⱼ₊½ = solver.mean(solver.α[idx], solver.α[ⱼ₊₁])
    θr_dτ_ⱼ₊½ = arithmetic_mean(θr_dτ[idx], θr_dτ[ⱼ₊₁])

    ∂H∂yⱼ₊½ = (solver.H[ⱼ₊₁] - solver.H[idx]) / dy

    qHyⱼ₊½[idx] = (qHyⱼ₊½[idx] * θr_dτ_ⱼ₊½ - αⱼ₊½ * ∂H∂yⱼ₊½) / (1 + θr_dτ_ⱼ₊½)

    qHy_2ⱼ₊½[idx] = -αⱼ₊½ * ∂H∂yⱼ₊½
  end

  return nothing
end

function compute_update!(solver::PseudoTransientSolver{2,T}, dt) where {T}

  #
  qHx = solver.qH.x
  qHy = solver.qH.y

  i, j = (1, 2)
  dx, dy = solver.spacing

  @inbounds for idx in solver.iterators.domain.cartesian
    ᵢ₋₁ = shift(idx, i, -1)
    ⱼ₋₁ = shift(idx, j, -1)

    ∇qH = (
      (qHx[idx] - qHx[ᵢ₋₁]) / dx + # ∂qH∂x
      (qHy[idx] - qHy[ⱼ₋₁]) / dy   # ∂qH∂y
    )

    solver.H[idx] = (
      (solver.H[idx] + solver.dτ_ρ[idx] * (solver.H_prev[idx] / dt - ∇qH)) /
      (1 + solver.dτ_ρ[idx] / dt)
    )
  end

  return nothing
end

function update_residual!(solver::PseudoTransientSolver{2,T}, dt) where {T}

  #
  qHx_2 = solver.qH_2.x
  qHy_2 = solver.qH_2.y

  i, j = (1, 2)
  dx, dy = solver.spacing

  for idx in solver.iterators.domain.cartesian
    ᵢ₋₁ = shift(idx, i, -1)
    ⱼ₋₁ = shift(idx, j, -1)

    ∇qH = (
      (qHx_2[idx] - qHx_2[ᵢ₋₁]) / dx + # ∂qH∂x
      (qHy_2[idx] - qHy_2[ⱼ₋₁]) / dy   # ∂qH∂y
    )

    solver.res[idx] = -(solver.H[idx] - solver.H_prev[idx]) / dt - ∇qH
  end

  return nothing
end

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
