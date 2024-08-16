
# ------------------------------------------------------------------------------------------
# 1D
# ------------------------------------------------------------------------------------------
function update_residuals_orthogonal_1d(
  solver::PseudoTransientSolver{N,T,BE}, mesh, Δt
) where {N,T,BE<:GPU}

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
    u, u_prev, ξx, qξ_ᵢ, qξ_ᵢ₋₁, source_term, Δt
  )

  return nothing
end

function _update_residual_1d_orthogonal_mesh!(u, u_prev, ξx, qξ_ᵢ, qξ_ᵢ₋₁, source_term, dt)
  ∇q = ξx^2 * (qξ_ᵢ - qξ_ᵢ₋₁)
  residuals = -(u - u_prev) / dt - ∇q + source_term
  return residuals
end

# ------------------------------------------------------------------------------------------
# 2D
# ------------------------------------------------------------------------------------------
function update_residuals_orthogonal_2d(
  solver::PseudoTransientSolver{N,T,BE}, mesh, Δt
) where {N,T,BE<:GPU}

  #
  iaxis, jaxis = (1, 2)
  domain = solver.iterators.domain.cartesian

  ᵢ₋₁ⱼ_domain = shift(domain, iaxis, -1)
  ᵢⱼ₋₁_domain = shift(domain, jaxis, -1)

  u = @view solver.u[domain]
  u_prev = @view solver.u_prev[domain]

  # note the q′ not q
  qξ_ᵢⱼ = @view solver.q′.x[domain]
  qη_ᵢⱼ = @view solver.q′.y[domain]
  qξ_ᵢ₋₁ⱼ = @view solver.q′.x[ᵢ₋₁ⱼ_domain]
  qη_ᵢⱼ₋₁ = @view solver.q′.y[ᵢⱼ₋₁_domain]

  ξx = @view mesh.cell_center_metrics.ξ.x₁[domain]
  ξy = @view mesh.cell_center_metrics.ξ.x₂[domain]
  ηx = @view mesh.cell_center_metrics.η.x₁[domain]
  ηy = @view mesh.cell_center_metrics.η.x₂[domain]

  source_term = @view solver.source_term[domain]
  residuals = @view solver.res[domain]

  @. residuals = _update_residual_2d_orthogonal_mesh!(
    u, u_prev, ξx, ξy, ηx, ηy, qξ_ᵢⱼ, qξ_ᵢ₋₁ⱼ, qη_ᵢⱼ, qη_ᵢⱼ₋₁, source_term, Δt
  )

  return nothing
end

function _update_residual_2d_orthogonal_mesh!(
  u, u_prev, ξx, ξy, ηx, ηy, qξ_ᵢⱼ, qξ_ᵢ₋₁ⱼ, qη_ᵢⱼ, qη_ᵢⱼ₋₁, source_term, dt
)
  ∂qξ∂ξ = (ξx^2 + ξy^2) * (qξ_ᵢⱼ - qξ_ᵢ₋₁ⱼ)
  ∂qη∂η = (ηx^2 + ηy^2) * (qη_ᵢⱼ - qη_ᵢⱼ₋₁)

  ∇q = ∂qξ∂ξ + ∂qη∂η

  residuals = -(u - u_prev) / dt - ∇q + source_term
  return residuals
end

# ------------------------------------------------------------------------------------------
# 3D
# ------------------------------------------------------------------------------------------

function update_residuals_orthogonal_3d(
  solver::PseudoTransientSolver{N,T,BE}, mesh, Δt
) where {N,T,BE<:GPU}

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
  )

  return nothing
end

function _update_residual_3d_orthogonal_mesh!(
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
  dt,
)
  ∂qξ∂ξ = (ξx^2 + ξy^2 + ξz^2) * (qξ_ᵢⱼₖ - qξ_ᵢ₋₁ⱼₖ)
  ∂qη∂η = (ηx^2 + ηy^2 + ηz^2) * (qη_ᵢⱼₖ - qη_ᵢⱼ₋₁ₖ)
  ∂qζ∂ζ = (ζx^2 + ζy^2 + ζz^2) * (qζ_ᵢⱼₖ - qζ_ᵢⱼₖ₋₁)

  ∇q = ∂qξ∂ξ + ∂qη∂η + ∂qζ∂ζ

  residuals = -(u - u_prev) / dt - ∇q + source_term
  return residuals
end