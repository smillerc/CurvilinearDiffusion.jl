
# ------------------------------------------------------------------------------------------
# 1D
# ------------------------------------------------------------------------------------------
function update_orthogonal!(
  solver::PseudoTransientSolver{1,T,BE}, mesh, Δt
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

  dτ_ρ = @view solver.dτ_ρ[domain]
  source_term = @view solver.source_term[domain]

  @. residuals = _update_1d_orthogonal_mesh!(
    u, u_prev, ξx, qξ_ᵢ, qξ_ᵢ₋₁, dτ_ρ, source_term, Δt
  )

  return nothing
end

function _update_1d_orthogonal_mesh!(u, u_prev, ξx, qξ_ᵢ, qξ_ᵢ₋₁, dτ_ρ, source_term, dt)
  ∇q = ξx^2 * (qξ_ᵢ - qξ_ᵢ₋₁)
  unew = (u + dτ_ρ * (u_prev / dt - ∇q + source_term)) / (1 + dτ_ρ / dt)
  return unew
end

# ------------------------------------------------------------------------------------------
# 2D
# ------------------------------------------------------------------------------------------
function update_orthogonal!(
  solver::PseudoTransientSolver{2,T,BE}, mesh, Δt
) where {T,BE<:GPU}

  #
  iaxis, jaxis = (1, 2)
  domain = solver.iterators.domain.cartesian

  ᵢ₋½_domain = shift(domain, iaxis, -1)
  ⱼ₋½_domain = shift(domain, jaxis, -1)

  u = @view solver.u[domain]
  u_prev = @view solver.u_prev[domain]

  # note the q′ not q
  qξᵢ₊½ = @view solver.q.x[domain]
  qηⱼ₊½ = @view solver.q.y[domain]
  qξᵢ₋½ = @view solver.q.x[ᵢ₋½_domain]
  qηⱼ₋½ = @view solver.q.y[ⱼ₋½_domain]

  ξx = @view mesh.cell_center_metrics.ξ.x₁[domain]
  ξy = @view mesh.cell_center_metrics.ξ.x₂[domain]
  ηx = @view mesh.cell_center_metrics.η.x₁[domain]
  ηy = @view mesh.cell_center_metrics.η.x₂[domain]

  source_term = @view solver.source_term[domain]

  source_term = @view solver.source_term[domain]
  dτ_ρ = @view solver.dτ_ρ[domain]

  @. u = _update_2d_orthogonal_mesh!(
    u, u_prev, ξx, ξy, ηx, ηy, qξᵢ₊½, qξᵢ₋½, qηⱼ₊½, qηⱼ₋½, dτ_ρ, source_term, Δt
  )

  return nothing
end

function _update_2d_orthogonal_mesh!(
  u, u_prev, ξx, ξy, ηx, ηy, qξᵢ₊½, qξᵢ₋½, qηⱼ₊½, qηⱼ₋½, dτ_ρ, source_term, dt
)
  ∂qξ∂ξ = (ξx^2 + ξy^2) * (qξᵢ₊½ - qξᵢ₋½)
  ∂qη∂η = (ηx^2 + ηy^2) * (qηⱼ₊½ - qηⱼ₋½)

  ∇q = ∂qξ∂ξ + ∂qη∂η

  unew = (u + dτ_ρ * (u_prev / dt - ∇q + source_term)) / (1 + dτ_ρ / dt)
  return unew
end

# ------------------------------------------------------------------------------------------
# 3D
# ------------------------------------------------------------------------------------------

function update_orthogonal!(
  solver::PseudoTransientSolver{3,T,BE}, mesh, Δt
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
  dτ_ρ = @view solver.dτ_ρ[domain]

  @. residuals = _update_3d_orthogonal_mesh!(
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
    dτ_ρ,
    source_term,
    Δt,
  )

  return nothing
end

function _update_3d_orthogonal_mesh!(
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
  dτ_ρ,
  source_term,
  dt,
)
  ∂qξ∂ξ = (ξx^2 + ξy^2 + ξz^2) * (qξ_ᵢⱼₖ - qξ_ᵢ₋₁ⱼₖ)
  ∂qη∂η = (ηx^2 + ηy^2 + ηz^2) * (qη_ᵢⱼₖ - qη_ᵢⱼ₋₁ₖ)
  ∂qζ∂ζ = (ζx^2 + ζy^2 + ζz^2) * (qζ_ᵢⱼₖ - qζ_ᵢⱼₖ₋₁)

  ∇q = ∂qξ∂ξ + ∂qη∂η + ∂qζ∂ζ

  unew = (u + dτ_ρ * (u_prev / dt - ∇q + source_term)) / (1 + dτ_ρ / dt)
  return unew
end