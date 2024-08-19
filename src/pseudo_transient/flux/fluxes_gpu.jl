
# ------------------------------------------------------------------------------------------
# 1D
# ------------------------------------------------------------------------------------------

function compute_flux!(
  solver::PseudoTransientSolver{1,T,BE}, ::CurvilinearGrid1D
) where {T,BE<:GPU}
  iaxis = 1

  ᵢ_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)

  ᵢ₊₁_domain = expand_upper(solver.iterators.domain.cartesian, iaxis, +1)

  qᵢ′ = @view solver.q′.x[ᵢ_domain]
  qᵢ = @view solver.q.x[ᵢ_domain]
  uᵢ = @view solver.u[ᵢ_domain]
  αᵢ = @view solver.α[ᵢ_domain]
  θr_dτᵢ = @view solver.θr_dτ[ᵢ_domain]
  uᵢ₊₁ = @view solver.u[ᵢ₊₁_domain]
  αᵢ₊₁ = @view solver.α[ᵢ₊₁_domain]
  θr_dτᵢ₊₁ = @view solver.θr_dτ[ᵢ₊₁_domain]

  @. qᵢ = flux_kernel!(qᵢ, uᵢ₊₁, uᵢ, αᵢ₊₁, αᵢ, θr_dτᵢ₊₁, θr_dτᵢ, solver.mean)
  @. qᵢ′ = fluxprime_kernel!(uᵢ₊₁, uᵢ, αᵢ₊₁, αᵢ, solver.mean)

  return nothing
end

# ------------------------------------------------------------------------------------------
# 2D
# ------------------------------------------------------------------------------------------

function compute_flux!(
  solver::PseudoTransientSolver{2,T,BE}, ::CurvilinearGrid2D
) where {T,BE<:GPU}
  iaxis, jaxis = (1, 2)

  ᵢ_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)

  ᵢ₊₁_domain = expand_upper(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊₁_domain = expand_upper(solver.iterators.domain.cartesian, jaxis, +1)

  qᵢ′ = @view solver.q′.x[ᵢ_domain]
  qᵢ = @view solver.q.x[ᵢ_domain]
  uᵢ = @view solver.u[ᵢ_domain]
  αᵢ = @view solver.α[ᵢ_domain]
  θr_dτᵢ = @view solver.θr_dτ[ᵢ_domain]
  uᵢ₊₁ = @view solver.u[ᵢ₊₁_domain]
  αᵢ₊₁ = @view solver.α[ᵢ₊₁_domain]
  θr_dτᵢ₊₁ = @view solver.θr_dτ[ᵢ₊₁_domain]

  @. qᵢ = flux_kernel!(qᵢ, uᵢ₊₁, uᵢ, αᵢ₊₁, αᵢ, θr_dτᵢ₊₁, θr_dτᵢ, solver.mean)
  @. qᵢ′ = fluxprime_kernel!(uᵢ₊₁, uᵢ, αᵢ₊₁, αᵢ, solver.mean)

  qⱼ′ = @view solver.q′.y[ⱼ_domain]
  qⱼ = @view solver.q.y[ⱼ_domain]
  uⱼ = @view solver.u[ⱼ_domain]
  αⱼ = @view solver.α[ⱼ_domain]
  θr_dτⱼ = @view solver.θr_dτ[ⱼ_domain]
  uⱼ₊₁ = @view solver.u[ⱼ₊₁_domain]
  αⱼ₊₁ = @view solver.α[ⱼ₊₁_domain]
  θr_dτⱼ₊₁ = @view solver.θr_dτ[ⱼ₊₁_domain]

  @. qⱼ = flux_kernel!(qⱼ, uⱼ₊₁, uⱼ, αⱼ₊₁, αⱼ, θr_dτⱼ₊₁, θr_dτⱼ, solver.mean)
  @. qⱼ′ = fluxprime_kernel!(uⱼ₊₁, uⱼ, αⱼ₊₁, αⱼ, solver.mean)

  return nothing
end

# ------------------------------------------------------------------------------------------
# 3D
# ------------------------------------------------------------------------------------------

function compute_flux!(
  solver::PseudoTransientSolver{3,T,BE}, ::CurvilinearGrid3D
) where {T,BE<:GPU}
  iaxis, jaxis, kaxis = (1, 2, 3)

  ᵢ_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)
  ₖ_domain = expand_lower(solver.iterators.domain.cartesian, kaxis, +1)

  ᵢ₊₁_domain = expand_upper(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊₁_domain = expand_upper(solver.iterators.domain.cartesian, jaxis, +1)
  ₖ₊₁_domain = expand_upper(solver.iterators.domain.cartesian, kaxis, +1)

  # ----------------------------------------------------
  qᵢ′ = @view solver.q′.x[ᵢ_domain]
  qᵢ = @view solver.q.x[ᵢ_domain]
  uᵢ = @view solver.u[ᵢ_domain]
  αᵢ = @view solver.α[ᵢ_domain]
  θr_dτᵢ = @view solver.θr_dτ[ᵢ_domain]
  uᵢ₊₁ = @view solver.u[ᵢ₊₁_domain]
  αᵢ₊₁ = @view solver.α[ᵢ₊₁_domain]
  θr_dτᵢ₊₁ = @view solver.θr_dτ[ᵢ₊₁_domain]

  @. qᵢ = flux_kernel!(qᵢ, uᵢ₊₁, uᵢ, αᵢ₊₁, αᵢ, θr_dτᵢ₊₁, θr_dτᵢ, solver.mean)
  @. qᵢ′ = fluxprime_kernel!(uᵢ₊₁, uᵢ, αᵢ₊₁, αᵢ, solver.mean)

  # ----------------------------------------------------
  qⱼ′ = @view solver.q′.y[ⱼ_domain]
  qⱼ = @view solver.q.y[ⱼ_domain]
  uⱼ = @view solver.u[ⱼ_domain]
  αⱼ = @view solver.α[ⱼ_domain]
  θr_dτⱼ = @view solver.θr_dτ[ⱼ_domain]
  uⱼ₊₁ = @view solver.u[ⱼ₊₁_domain]
  αⱼ₊₁ = @view solver.α[ⱼ₊₁_domain]
  θr_dτⱼ₊₁ = @view solver.θr_dτ[ⱼ₊₁_domain]

  @. qⱼ = flux_kernel!(qⱼ, uⱼ₊₁, uⱼ, αⱼ₊₁, αⱼ, θr_dτⱼ₊₁, θr_dτⱼ, solver.mean)
  @. qⱼ′ = fluxprime_kernel!(uⱼ₊₁, uⱼ, αⱼ₊₁, αⱼ, solver.mean)

  # ----------------------------------------------------
  qₖ′ = @view solver.q′.z[ₖ_domain]
  qₖ = @view solver.q.z[ₖ_domain]
  uₖ = @view solver.u[ₖ_domain]
  αₖ = @view solver.α[ₖ_domain]
  θr_dτₖ = @view solver.θr_dτ[ₖ_domain]
  uₖ₊₁ = @view solver.u[ₖ₊₁_domain]
  αₖ₊₁ = @view solver.α[ₖ₊₁_domain]
  θr_dτₖ₊₁ = @view solver.θr_dτ[ₖ₊₁_domain]

  @. qₖ = flux_kernel!(qₖ, uₖ₊₁, uₖ, αₖ₊₁, αₖ, θr_dτₖ₊₁, θr_dτₖ, solver.mean)
  @. qₖ′ = fluxprime_kernel!(uₖ₊₁, uₖ, αₖ₊₁, αₖ, solver.mean)

  return nothing
end