
# ------------------------------------------------------------------------------------------
# 1D
# ------------------------------------------------------------------------------------------
function update_residuals_orthogonal!(
  solver::PseudoTransientSolver{1,T,BE}, mesh, Δt, ϵ=eps(T)
) where {T,BE<:GPU}

  #
  iaxis = 1
  domain = solver.iterators.domain.cartesian

  ᵢ₋₁ⱼ_domain = shift(domain, iaxis, -1)

  u = @view solver.u[domain]
  u_prev = @view solver.u_prev[domain]

  # note the q′ not q
  qξ_ᵢ = @view solver.q′.x[domain]
  qξ_ᵢ₋₁ = @view solver.q′.x[ᵢ₋₁ⱼ_domain]

  ξx = @view mesh.cell_center_metrics.ξ.x₁[domain]

  source_term = @view solver.source_term[domain]
  residuals = @view solver.res[domain]

  @. residuals = _update_residual_1d_orthogonal_mesh!(
    u, u_prev, ξx, qξ_ᵢ, qξ_ᵢ₋₁, source_term, Δt, ϵ
  )

  return nothing
end

@inline function _update_residual_1d_orthogonal_mesh!(
  u::T, u_prev, ξx, qξᵢ₊½, qξᵢ₋½, source_term, dt, ϵ
) where {T}
  #
  ∇q = flux_divergence_orth(qξᵢ₊½, qξᵢ₋½, ξx)

  du = u - u_prev
  du = du * !isapprox(u, u_prev; rtol=ϵ)

  residuals = -du / dt - ∇q + source_term
  return residuals
end

# ------------------------------------------------------------------------------------------
# 2D
# ------------------------------------------------------------------------------------------
function update_residuals_orthogonal!(
  solver::PseudoTransientSolver{2,T,BE}, mesh, Δt, ϵ=eps(T)
) where {T,BE<:GPU}

  #
  iaxis, jaxis = (1, 2)
  domain = solver.iterators.domain.cartesian

  ᵢ₋½_domain = shift(domain, iaxis, -1)
  ⱼ₋½_domain = shift(domain, jaxis, -1)

  u = @view solver.u[domain]
  u_prev = @view solver.u_prev[domain]

  # note the q′ not q
  qξᵢ₊½ = @view solver.q′.x[domain]
  qηⱼ₊½ = @view solver.q′.y[domain]
  qξᵢ₋½ = @view solver.q′.x[ᵢ₋½_domain]
  qηⱼ₋½ = @view solver.q′.y[ⱼ₋½_domain]

  ξx = @view mesh.cell_center_metrics.ξ.x₁[domain]
  ξy = @view mesh.cell_center_metrics.ξ.x₂[domain]
  ηx = @view mesh.cell_center_metrics.η.x₁[domain]
  ηy = @view mesh.cell_center_metrics.η.x₂[domain]

  source_term = @view solver.source_term[domain]
  residuals = @view solver.res[domain]

  @. residuals = _update_residual_2d_orthogonal_mesh!(
    u, u_prev, ξx, ξy, ηx, ηy, qξᵢ₊½, qξᵢ₋½, qηⱼ₊½, qηⱼ₋½, source_term, Δt, ϵ
  )

  return nothing
end

@inline function _update_residual_2d_orthogonal_mesh!(
  u, u_prev, ξx, ξy, ηx, ηy, qξᵢ₊½, qξᵢ₋½, qηⱼ₊½, qηⱼ₋½, source_term, dt, ϵ
)

  #
  ∇q = flux_divergence_orth(qξᵢ₊½, qξᵢ₋½, qηⱼ₊½, qηⱼ₋½, ξx, ξy, ηx, ηy)

  du = u - u_prev
  du = du * !isapprox(u, u_prev; rtol=ϵ)

  residuals = -du / dt - ∇q + source_term
  return residuals
end

# ------------------------------------------------------------------------------------------
# 3D
# ------------------------------------------------------------------------------------------

function update_residuals_orthogonal!(
  solver::PseudoTransientSolver{3,T,BE}, mesh, Δt, ϵ=eps(T)
) where {T,BE<:GPU}

  #
  iaxis, jaxis, kaxis = (1, 2, 3)
  domain = solver.iterators.domain.cartesian

  ᵢ₋₁ⱼₖ_domain = shift(domain, iaxis, -1)
  ᵢⱼ₋₁ₖ_domain = shift(domain, jaxis, -1)
  ᵢⱼₖ₋₁_domain = shift(domain, kaxis, -1)

  u = @view solver.u[domain]
  u_prev = @view solver.u_prev[domain]

  # note the q′ not q
  qξ_ᵢⱼₖ = @view solver.q′.x[domain]
  qξ_ᵢ₋₁ⱼₖ = @view solver.q′.x[ᵢ₋₁ⱼₖ_domain]
  qη_ᵢⱼₖ = @view solver.q′.y[domain]
  qη_ᵢⱼ₋₁ₖ = @view solver.q′.y[ᵢⱼ₋₁ₖ_domain]
  qζ_ᵢⱼₖ = @view solver.q′.z[domain]
  qζ_ᵢⱼₖ₋₁ = @view solver.q′.z[ᵢⱼₖ₋₁_domain]

  ξx = @view mesh.cell_center_metrics.ξ.x₁[domain]
  ξy = @view mesh.cell_center_metrics.ξ.x₂[domain]
  ξz = @view mesh.cell_center_metrics.ξ.x₃[domain]

  ηx = @view mesh.cell_center_metrics.η.x₁[domain]
  ηy = @view mesh.cell_center_metrics.η.x₂[domain]
  ηz = @view mesh.cell_center_metrics.η.x₃[domain]

  ζx = @view mesh.cell_center_metrics.η.x₁[domain]
  ζy = @view mesh.cell_center_metrics.η.x₂[domain]
  ζz = @view mesh.cell_center_metrics.η.x₃[domain]

  source_term = @view solver.source_term[domain]
  residuals = @view solver.res[domain]

  @. residuals = _update_residual_3d_orthogonal_mesh!(
    u,
    u_prev,
    ξx,
    ξy,
    ξz,
    ηx,
    ηy,
    ηz,
    ζx,
    ζy,
    ζz,
    qξ_ᵢⱼₖ,
    qξ_ᵢ₋₁ⱼₖ,
    qη_ᵢⱼₖ,
    qη_ᵢⱼ₋₁ₖ,
    qζ_ᵢⱼₖ,
    qζ_ᵢⱼₖ₋₁,
    source_term,
    Δt,
    ϵ,
  )

  return nothing
end

@inline function _update_residual_3d_orthogonal_mesh!(
  u,
  u_prev,
  ξx,
  ξy,
  ξz,
  ηx,
  ηy,
  ηz,
  ζx,
  ζy,
  ζz,
  qξᵢ₊½,
  qξᵢ₋½,
  qηⱼ₊½,
  qηⱼ₋½,
  qζₖ₊½,
  qζₖ₋½,
  source_term,
  dt,
  ϵ,
)

  #
  ∇q = flux_divergence_orth(
    qξᵢ₊½, qξᵢ₋½, qηⱼ₊½, qηⱼ₋½, qζₖ₊½, qζₖ₋½, ξx, ξy, ξz, ηx, ηy, ηz, ζx, ζy, ζz
  )

  du = u - u_prev
  du = du * !isapprox(u, u_prev; rtol=ϵ)

  residuals = -du / dt - ∇q + source_term
  return residuals
end