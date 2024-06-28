module PseudoTransientScheme

using LinearAlgebra: norm

using CartesianDomains: expand, shift, expand_lower
using CurvilinearGrids: CurvilinearGrid2D, cellsize_withhalo, coords
using KernelAbstractions: CPU
using Polyester: @batch

using ..BoundaryConditions

export PseudoTransientSolver

struct PseudoTransientSolver{N,T,AA<:AbstractArray{T,N},NT1,DM,B}
  H::AA
  H_prev::AA
  source_term::AA
  qH::NT1
  qH_2::NT1
  res::AA
  Re::AA
  D::AA # diffusivity
  θr_dτ::AA
  dτ_ρ::AA
  spacing::NTuple{N,T}
  L::T
  domain::DM
  bcs::B # boundary conditions
end

function PseudoTransientSolver(
  mesh::CurvilinearGrid2D, bcs; backend=CPU(), face_diffusivity=:harmonic
)
  #
  #         H
  #     |--------|--------|--------|--------|
  #           q_i+1/2
  #

  ni, nj = cellsize_withhalo(mesh)

  # cell-based
  H = zeros(ni, nj)
  H_prev = zeros(ni, nj)
  S = zeros(ni, nj) # source term

  # edge-based
  qH = (x=zeros(ni, nj), y=zeros(ni, nj))
  qH² = (x=zeros(ni, nj), y=zeros(ni, nj))
  res = zeros(ni, nj)
  Re = zeros(ni, nj)
  D = ones(ni, nj)
  θr_dτ = zeros(ni, nj)
  dτ_ρ = zeros(ni, nj)

  x, y = coords(mesh)
  spacing = (
    round(minimum(diff(x; dims=1)); sigdigits=10),
    round(minimum(diff(y; dims=2)); sigdigits=10),
  )
  min_x, max_x = extrema(x)
  min_y, max_y = extrema(y)
  L = max(abs(max_x - min_x), abs(max_y - min_y))
  domain = mesh.iterators.cell.domain

  return PseudoTransientSolver(
    H, H_prev, S, qH, qH², res, Re, D, θr_dτ, dτ_ρ, spacing, L, domain, bcs
  )
end

# solve a single time-step dt
function solve!(
  solver::PseudoTransientSolver{N,T},
  mesh,
  dt;
  max_iter=1e5,
  tol=1e-8,
  error_check_interval=10,
) where {N,T}

  #
  iter = 0
  err = 2 * tol

  CFL = 1 / sqrt(N)

  dx, dy = solver.spacing
  Vpdτ = CFL * min(dx, dy)

  copy!(solver.H_prev, solver.H)

  # Pseudo-transient iteration
  while err > tol && iter < max_iter
    @info "Iter: $iter"
    iter += 1

    @show extrema(solver.H_prev[solver.domain])
    @show extrema(solver.H[solver.domain])
    # Diffusion coefficient
    # update_conductivity!(solver, mesh, T, ρ, cₚ, κ)
    applybcs!(solver.bcs, mesh, solver.H)

    update_iteration_params!(solver, Vpdτ, dt)
    # @show extrema(Vpdτ)
    # @show extrema(solver.D[solver.domain])
    # @show extrema(solver.Re[solver.domain])
    # @show extrema(solver.θr_dτ[solver.domain])
    # @show extrema(solver.dτ_ρ[solver.domain])

    # @show extrema(solver.H[solver.domain])
    # error("done")

    applybcs!(solver.bcs, mesh, solver.D)
    applybcs!(solver.bcs, mesh, solver.dτ_ρ)
    applybcs!(solver.bcs, mesh, solver.θr_dτ)

    compute_flux!(solver)
    compute_update!(solver, dt)

    # Check error (explicit residual)
    if iter % error_check_interval == 0
      update_residual!(solver, dt)

      res = @view solver.res[solver.domain]
      err = norm(res) / sqrt(length(res))

      @show err norm(res) sqrt(length(res))
    end
  end

  copy!(solver.H_prev, solver.H)

  if iter == max_iter
    @error(
      "Maximum iteration limit reached ($max_iter), current error is $(err), tolerance is $tol, exiting...",
    )
  end

  return err, iter
end

function update_iteration_params!(
  solver,
  Vpdτ,
  dt,
  # β=1 / 1.2, # iteration scaling parameter
  β=1, # iteration scaling parameter
)
  for idx in solver.domain
    solver.Re[idx] = π + sqrt(π^2 + (solver.L^2 / solver.D[idx] / dt))
  end

  for idx in solver.domain
    solver.dτ_ρ[idx] = Vpdτ * solver.L / solver.D[idx] / solver.Re[idx] * β
  end

  for idx in solver.domain
    solver.θr_dτ[idx] = solver.L / Vpdτ / solver.Re[idx] * β
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

  ᵢ₊½_domain = expand_lower(solver.domain, i, +1)
  @show solver.domain
  @show ᵢ₊½_domain
  for idx in ᵢ₊½_domain
    ᵢ₊₁ = shift(idx, i, +1)
    αᵢ₊½ = (solver.D[idx] + solver.D[ᵢ₊₁]) / 2
    θr_dτ_ᵢ₊½ = (θr_dτ[idx] + θr_dτ[ᵢ₊₁]) / 2
    ∂H∂xᵢ₊½ = (solver.H[ᵢ₊₁] - solver.H[idx]) / dx
    qHxᵢ₊½[idx] = (qHxᵢ₊½[idx] * θr_dτ_ᵢ₊½ - αᵢ₊½ * ∂H∂xᵢ₊½) / (1 + θr_dτ_ᵢ₊½)
    qHx_2ᵢ₊½[idx] = -αᵢ₊½ * ∂H∂xᵢ₊½
  end

  ⱼ₊½_domain = expand_lower(solver.domain, j, +1)
  for idx in ⱼ₊½_domain
    ⱼ₊₁ = shift(idx, j, +1)
    αⱼ₊½ = (solver.D[idx] + solver.D[ⱼ₊₁]) / 2
    θr_dτ_ⱼ₊½ = (θr_dτ[idx] + θr_dτ[ⱼ₊₁]) / 2
    ∂H∂yⱼ₊½ = (solver.H[ⱼ₊₁] - solver.H[idx]) / dy

    qHyⱼ₊½[idx] = (qHyⱼ₊½[idx] * θr_dτ_ⱼ₊½ - αⱼ₊½ * ∂H∂yⱼ₊½) / (1 + θr_dτ_ⱼ₊½)

    qHy_2ⱼ₊½[idx] = -αⱼ₊½ * ∂H∂yⱼ₊½
  end

  @show extrema(view(qHxᵢ₊½, ᵢ₊½_domain))
  @show extrema(view(qHyⱼ₊½, ⱼ₊½_domain))
  @show extrema(view(qHx_2ᵢ₊½, ᵢ₊½_domain))
  @show extrema(view(qHy_2ⱼ₊½, ⱼ₊½_domain))

  # @show qHx[solver.domain][49, 50]
  return nothing
end

function compute_update!(solver::PseudoTransientSolver{2,T}, dt) where {T}

  #
  qHx = solver.qH.x
  qHy = solver.qH.y

  i, j = (1, 2)
  dx, dy = solver.spacing

  for idx in solver.domain
    ᵢ₋₁ = shift(idx, i, -1)
    ⱼ₋₁ = shift(idx, j, -1)
    ᵢ₊₁ = shift(idx, i, +1)
    ⱼ₊₁ = shift(idx, j, +1)

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

  for idx in solver.domain
    ᵢ₋₁ = shift(idx, i, -1)
    ⱼ₋₁ = shift(idx, j, -1)
    ᵢ₊₁ = shift(idx, i, +1)
    ⱼ₊₁ = shift(idx, j, +1)

    ∇qH = (
      (qHx_2[idx] - qHx_2[ᵢ₋₁]) / dx + # ∂qH∂x
      (qHy_2[idx] - qHy_2[ⱼ₋₁]) / dy   # ∂qH∂y
    )

    solver.res[idx] = -(solver.H[idx] - solver.H_prev[idx]) / dt - ∇qH
  end

  return nothing
end

end # module
