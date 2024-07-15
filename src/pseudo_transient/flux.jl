@kernel function flux_kernel!(
  qᵢ₊½, q′ᵢ₊½, u, α, θr_dτ, axis, index_offset, mean_func::F
) where {F}
  idx = @index(Global, Cartesian)
  ᵢ = idx + index_offset

  @inbounds begin
    ᵢ₊₁ = shift(ᵢ, axis, +1)

    # edge diffusivity / iter params
    αᵢ₊½ = mean_func(α[ᵢ], α[ᵢ₊₁])
    θr_dτ_ᵢ₊½ = mean_func(θr_dτ[ᵢ], θr_dτ[ᵢ₊₁])
    # θr_dτ_ᵢ₊½ = (θr_dτ[ᵢ] + θr_dτ[ᵢ₊₁]) / 2
    # θr_dτ_ᵢ₊½ = max(θr_dτ[ᵢ], θr_dτ[ᵢ₊₁])

    du = (u[ᵢ₊₁] - u[ᵢ])
    du = du * (abs(du) >= 1e-14)
    _qᵢ₊½ = -αᵢ₊½ * du * isfinite(du)

    qᵢ₊½[ᵢ] = (qᵢ₊½[ᵢ] * θr_dτ_ᵢ₊½ + _qᵢ₊½) / (1 + θr_dτ_ᵢ₊½)
    q′ᵢ₊½[ᵢ] = _qᵢ₊½
  end
end

function compute_flux!(solver::PseudoTransientSolver{2,T}, ::CurvilinearGrid2D) where {T}
  iaxis = 1
  jaxis = 2
  ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)

  ᵢ₊½_idx_offset = first(ᵢ₊½_domain) - oneunit(first(ᵢ₊½_domain))
  ⱼ₊½_idx_offset = first(ⱼ₊½_domain) - oneunit(first(ⱼ₊½_domain))

  flux_kernel!(solver.backend)(
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

  flux_kernel!(solver.backend)(
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

function compute_flux!(solver::PseudoTransientSolver{3,T}, ::CurvilinearGrid3D) where {T}
  iaxis = 1
  jaxis = 2
  kaxis = 3

  ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)
  ₖ₊½_domain = expand_lower(solver.iterators.domain.cartesian, kaxis, +1)

  ᵢ₊½_idx_offset = first(ᵢ₊½_domain) - oneunit(first(ᵢ₊½_domain))
  ⱼ₊½_idx_offset = first(ⱼ₊½_domain) - oneunit(first(ⱼ₊½_domain))
  ₖ₊½_idx_offset = first(ₖ₊½_domain) - oneunit(first(ₖ₊½_domain))

  flux_kernel!(solver.backend)(
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

  flux_kernel!(solver.backend)(
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

  flux_kernel!(solver.backend)(
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