
@kernel function flux_kernel!(
  qᵢ₊½::AbstractArray{T,2}, q′ᵢ₊½::AbstractArray{T,2}, u, α, θr_dτ, axis, I0, mean_func::F
) where {T,F}
  idx = @index(Global, Cartesian)
  idx += I0

  @inbounds begin
    ᵢ₊₁ = shift(idx, axis, +1)

    # edge diffusivity / iter params
    αᵢ₊½ = mean_func(α[idx], α[ᵢ₊₁])
    θr_dτ_ᵢ₊½ = mean_func(θr_dτ[idx], θr_dτ[ᵢ₊₁])

    _qᵢ₊½ = -αᵢ₊½ * (u[ᵢ₊₁] - u[idx])

    qᵢ₊½[idx] = (qᵢ₊½[idx] * θr_dτ_ᵢ₊½ + _qᵢ₊½) / (1 + θr_dτ_ᵢ₊½)
    q′ᵢ₊½[idx] = _qᵢ₊½
  end
end

function compute_flux!(solver::PseudoTransientSolver{2,T}, ::CurvilinearGrid2D) where {T}
  iaxis, jaxis = (1, 2)

  ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)

  # domain = solver.iterators.domain.cartesian
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
