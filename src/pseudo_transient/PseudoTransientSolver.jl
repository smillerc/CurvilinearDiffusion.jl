module PseudoTransientScheme

using CartesianDomains
using CurvilinearGrids

struct PseudoTransientSolver{N,T,AA<:AbstractArray{T,N},NT1}
  H::AA
  H_old::AA
  qH::NT1
  qH_2::NT1
  res::AA
  Re::AA
  α::AA
  θr_dτ::AA
  dτ_ρ::AA
end

function PseudoTransientSolver()
  #
  #         H
  #     |--------|--------|--------|--------|
  #           q_i+1/2
  #

  # cell-based
  H = zeros(ni)
  H_old = zeros(ni)

  # edge-based
  qH = (x = zeros(ni))
  qH² = (x = zeros(ni))
  res = zeros(ni)
  Re = zeros(ni)
  α = zeros(ni)
  θr_dτ = zeros(ni)
  dτ_ρ = zeros(ni)

  return PseudoTransientSolver(H, H_old, qH, qH², res, Re, α, θr_dτ, dτ_ρ)
end

function solve!(solver, mesh, u, dt)
  iter = 0
  err = 2 * tol

  # Pseudo-transient iteration
  while err > tol && iter < itMax
    # Diffusion coefficient
    # D .= H .^ 3
    update_conductivity!(solver, mesh, T, ρ, cₚ, κ)

    # Assign iter params
    @. Re = π + sqrt(π^2 + lx^2 / max(D, ε) / dt)

    # θr_dτ -> θr / dτ
    θr_dτ .= lx ./ Vpdτ ./ av(Re)

    # dτ_ρ -> dτ /ρ

    dτ_ρ .= Vpdτ .* lx ./ max.(D[2:(end - 1)], ε) ./ Re[2:(end - 1)]

    # PT updates
    # if implicit
    #   qHx .= (qHx .* θr_dτ .- av(D) .* diff(H) ./ dx) ./ (1.0 .+ θr_dτ)
    #   H[2:(end - 1)] .=
    #     (H[2:(end - 1)] .+ dτ_ρ .* (Hold[2:(end - 1)] ./ dt .- diff(qHx) ./ dx)) ./
    #     (1.0 .+ dτ_ρ ./ dt)
    # else
    @. qHx = qHx + 1 / θr_dτ * (-qHx - av(D) * diff(H) / dx)

    H[2:(end - 1)] .=
      H[2:(end - 1)] .+
      dτ_ρ .* (.-(H[2:(end - 1)] .- Hold[2:(end - 1)]) ./ dt .- diff(qHx) ./ dx)
    # end

    iter += 1
    # Check error (explicit residual)
    if iter % nout == 0
      qHx2 .= .-av(D) .* diff(H) ./ dx
      ResH .= .-(H[2:(end - 1)] .- Hold[2:(end - 1)]) ./ dt .- diff(qHx2) ./ dx
      err = norm(ResH) / sqrt(length(ResH))
    end
  end

  # Update H
  Hold .= H
  ittot += iter
  it += 1
  t += dt

  return nothing
end

function update_iteration_params!(θr_dτ, dτ_ρ, Vpdτ)
  # θr_dτ -> θr / dτ
  θr_dτ .= lx ./ Vpdτ ./ av(Re)

  # dτ_ρ -> dτ /ρ

  dτ_ρ .= Vpdτ .* lx ./ max.(D[2:(end - 1)], ε) ./ Re[2:(end - 1)]
end

function update_flux!(solver::PseudoTransientSolver{2,T}, Hᵏ) where {T}
  qHx = solver.qH.x
  qHy = solver.qH.y

  θr_dτ = solver.θr_dτ
  qHx_2 = solver.qH_2.x
  qHy_2 = solver.qH_2.y

  # fluxes are on the edge, H_i+1/2
  for idx in domain
    ᵢ₊₁ = shift(idx, 1, +1)
    ⱼ₊₁ = shift(idx, 2, +1)

    αᵢ₊½ = 0.5(α[idx] + α[ᵢ₊₁])
    αⱼ₊½ = 0.5(α[idx] + α[ⱼ₊₁])

    ∂H∂x = (Hᵏ[ᵢ₊₁] - Hᵏ[idx]) / dx
    ∂H∂y = (Hᵏ[ⱼ₊₁] - Hᵏ[idx]) / dy

    qHx[idx] = (qHx[idx] * θr_dτ[idx] - αᵢ₊½ * ∂H∂x) / (1.0 + θr_dτ[idx])
    qHy[idx] = (qHy[idx] * θr_dτ[idx] - αⱼ₊½ * ∂H∂y) / (1.0 + θr_dτ[idx])
    qHx_2[idx] = -αᵢ₊½ * ∂H∂x
    qHy_2[idx] = -αⱼ₊½ * ∂H∂y
  end

  return nothing
end

function compute_update!(solver::PseudoTransientSolver{2,T}, H, H_prev, dt) where {T}

  #

  qHx = solver.qH.x
  qHy = solver.qH.y

  for idx in domain
    ∂qH∂x = (qHx[ᵢ₊₁] - qHx[idx]) / dx
    ∂qH∂y = (qHy[ⱼ₊₁] - qHy[idx]) / dy

    ∇qH = ∂qH∂x + ∂qH∂y
    H[idx] = (H[idx] + dτ_ρ[idx] * (H_prev[idx] / dt - ∇qH)) / (1.0 + dτ_ρ[idx] / dt)
  end

  return nothing
end

function check_residual() end

end # module
