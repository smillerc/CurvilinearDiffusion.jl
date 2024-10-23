
function _cpu_flux_kernel!(
  qᵢ₊½::AbstractArray{T,N}, q′ᵢ₊½, u, α, θr_dτ, axis, domain, mean_func::F
) where {T,N,F}
  #

  @batch for idx in domain
    ᵢ₊₁ = shift(idx, axis, +1)

    uᵢ₊₁ = u[ᵢ₊₁]
    uᵢ = u[idx]
    _qᵢ₊½ = qᵢ₊½[idx] # current flux value

    αᵢ₊₁ = α[ᵢ₊₁]
    αᵢ = α[idx]
    θr_dτᵢ₊₁ = θr_dτ[ᵢ₊₁]
    θr_dτᵢ = θr_dτ[idx]

    # update the flux w/ intertial terms
    @inline qᵢ₊½[idx] = flux_kernel!(_qᵢ₊½, uᵢ₊₁, uᵢ, αᵢ₊₁, αᵢ, θr_dτᵢ₊₁, θr_dτᵢ, mean_func)

    # and the "plain" flux
    @inline q′ᵢ₊½[idx] = fluxprime_kernel!(uᵢ₊₁, uᵢ, αᵢ₊₁, αᵢ, mean_func)
  end

  return nothing
end

# ------------------------------------------------------------------------------------------
# 2D
# ------------------------------------------------------------------------------------------

function compute_flux!(
  solver::PseudoTransientSolver{2,T,BE}, ::CurvilinearGrid2D
) where {T,BE<:CPU}

  #
  iaxis, jaxis = (1, 2)

  ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)

  _cpu_flux_kernel!(
    solver.q.x,
    solver.q′.x,
    solver.u,
    solver.α,
    solver.θr_dτ,
    iaxis,
    ᵢ₊½_domain,
    solver.mean;
  )

  _cpu_flux_kernel!(
    solver.q.y,
    solver.q′.y,
    solver.u,
    solver.α,
    solver.θr_dτ,
    jaxis,
    ⱼ₊½_domain,
    solver.mean;
  )

  return nothing
end

# ------------------------------------------------------------------------------------------
# 3D
# ------------------------------------------------------------------------------------------

function compute_flux!(
  solver::PseudoTransientSolver{3,T,BE}, ::CurvilinearGrid3D
) where {T,BE<:CPU}
  iaxis, jaxis, kaxis = (1, 2, 3)

  ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)
  ₖ₊½_domain = expand_lower(solver.iterators.domain.cartesian, kaxis, +1)

  _cpu_flux_kernel!(
    solver.q.x,
    solver.q′.x,
    solver.u,
    solver.α,
    solver.θr_dτ,
    iaxis,
    ᵢ₊½_domain,
    solver.mean;
  )

  _cpu_flux_kernel!(
    solver.q.y,
    solver.q′.y,
    solver.u,
    solver.α,
    solver.θr_dτ,
    jaxis,
    ⱼ₊½_domain,
    solver.mean;
  )

  _cpu_flux_kernel!(
    solver.q.z,
    solver.q′.z,
    solver.u,
    solver.α,
    solver.θr_dτ,
    kaxis,
    ₖ₊½_domain,
    solver.mean;
  )

  return nothing
end