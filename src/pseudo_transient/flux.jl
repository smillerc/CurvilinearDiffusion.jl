
@kernel inbounds = true function _flux_kernel!(
  qᵢ₊½::AbstractArray{T,N}, q′ᵢ₊½, u, α, θr_dτ, axis, I0, mean_func::F
) where {T,N,F}

  # get the global index and offset for the inner domain
  idx = @index(Global, Cartesian)
  idx += I0

  # ϵ = eps(T)
  ᵢ₊₁ = shift(idx, axis, +1)
  # ᵢ₊₂ = shift(idx, axis, +2)
  # ᵢ₊₃ = shift(idx, axis, +3)
  # ᵢ₋₁ = shift(idx, axis, -1)
  # ᵢ₋₂ = shift(idx, axis, -2)

  # edge diffusivity / iter params
  @inline αᵢ₊½ = mean_func(α[idx], α[ᵢ₊₁])
  @inline θr_dτ_ᵢ₊½ = mean_func(θr_dτ[idx], θr_dτ[ᵢ₊₁]) # do NOT use max here, or it will fail to converge

  # f1 = 1 / 48
  # f2 = 1 / 16
  # f3 = 7 / 24

  du = u[ᵢ₊₁] - u[idx] # 2nd order
  # du_r = (u[ᵢ₊₂] - u[idx]) / 2
  # du_l = (u[ᵢ₊₁] - u[ᵢ₋₁]) / 2
  # du = (du_l + du_r) / 2
  # du = ( # 4th order
  #   -(7 / 24) * u[idx]  #
  #   + (7 / 24) * u[ᵢ₊₁]  #
  #   - (1 / 16) * u[ᵢ₊₂]  #
  #   - (1 / 48) * u[ᵢ₊₃]  #
  #   +
  #   (1 / 16) * u[ᵢ₋₁]  #
  #   +
  #   (1 / 48) * u[ᵢ₋₂] #
  # )
  # du = ( # 4th order
  #   -u[idx]  #
  #   + (1 / 3) * u[ᵢ₊₁]  #
  #   + (1 / 2) * u[ᵢ₊₂]  #
  #   - (1 / 20) * u[ᵢ₊₃]  #
  #   + (1 / 4) * u[ᵢ₋₁]  #
  #   - (1 / 30) * u[ᵢ₋₂] #
  # )
  # du = du * !isapprox(u[ᵢ₊₁], u[idx]) # epsilon check
  # du = du * (abs(du) >= ϵ) # epsilon check

  _qᵢ₊½ = -αᵢ₊½ * du

  qᵢ₊½[idx] = (qᵢ₊½[idx] * θr_dτ_ᵢ₊½ + _qᵢ₊½) / (1 + θr_dτ_ᵢ₊½)
  q′ᵢ₊½[idx] = _qᵢ₊½
end

# 2D 

function compute_flux!(solver::PseudoTransientSolver{2,T}, ::CurvilinearGrid2D) where {T}
  iaxis, jaxis = (1, 2)

  ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)

  # domain = solver.iterators.domain.cartesian
  ᵢ₊½_idx_offset = first(ᵢ₊½_domain) - oneunit(first(ᵢ₊½_domain))
  ⱼ₊½_idx_offset = first(ⱼ₊½_domain) - oneunit(first(ⱼ₊½_domain))

  _flux_kernel!(solver.backend)(
    solver.q.x,
    solver.q′.x,
    solver.u,
    solver.α,
    solver.θr_dτ,
    iaxis,
    ᵢ₊½_idx_offset,
    solver.mean;
    ndrange=size(ᵢ₊½_domain),
  )

  _flux_kernel!(solver.backend)(
    solver.q.y,
    solver.q′.y,
    solver.u,
    solver.α,
    solver.θr_dτ,
    jaxis,
    ⱼ₊½_idx_offset,
    solver.mean;
    ndrange=size(ⱼ₊½_domain),
  )

  KernelAbstractions.synchronize(solver.backend)

  return nothing
end

# 3D 

function compute_flux!(solver::PseudoTransientSolver{3,T}, ::CurvilinearGrid3D) where {T}
  iaxis, jaxis, kaxis = (1, 2, 3)

  ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)
  ₖ₊½_domain = expand_lower(solver.iterators.domain.cartesian, kaxis, +1)

  ᵢ₊½_idx_offset = first(ᵢ₊½_domain) - oneunit(first(ᵢ₊½_domain))
  ⱼ₊½_idx_offset = first(ⱼ₊½_domain) - oneunit(first(ⱼ₊½_domain))
  ₖ₊½_idx_offset = first(ₖ₊½_domain) - oneunit(first(ₖ₊½_domain))

  _flux_kernel!(solver.backend)(
    solver.q.x,
    solver.q′.x,
    solver.u,
    solver.α,
    solver.θr_dτ,
    iaxis,
    ᵢ₊½_idx_offset,
    solver.mean;
    ndrange=size(ᵢ₊½_domain),
  )

  _flux_kernel!(solver.backend)(
    solver.q.y,
    solver.q′.y,
    solver.u,
    solver.α,
    solver.θr_dτ,
    jaxis,
    ⱼ₊½_idx_offset,
    solver.mean;
    ndrange=size(ⱼ₊½_domain),
  )

  _flux_kernel!(solver.backend)(
    solver.q.z,
    solver.q′.z,
    solver.u,
    solver.α,
    solver.θr_dτ,
    kaxis,
    ₖ₊½_idx_offset,
    solver.mean;
    ndrange=size(ₖ₊½_domain),
  )

  KernelAbstractions.synchronize(solver.backend)

  return nothing
end