# ------------------------------------------------------------------------------------------
# 1D
# ------------------------------------------------------------------------------------------

function update_orthogonal!(
  solver::PseudoTransientSolver{1,T,BE}, mesh, Δt
) where {T,BE<:CPU}

  #
  domain = solver.iterators.domain.cartesian

  u = solver.u
  u_prev = solver.u_prev

  qξ = solver.q.x
  dτ_ρ = solver.dτ_ρ
  source_term = solver.source_term

  ξ_x = mesh.cell_center_metrics.ξ.x₁

  @batch for idx in domain
    i, = idx.I
    ∇q = (ξ_x[idx]^2) * (qξ[idx] - qξ[i - 1])

    u[idx] = (
      (u[idx] + dτ_ρ[idx] * (u_prev[idx] / Δt - ∇q + source_term[idx])) /
      (1 + dτ_ρ[idx] / Δt)
    )
  end

  return nothing
end

# ------------------------------------------------------------------------------------------
# 2D
# ------------------------------------------------------------------------------------------
function update_orthogonal!(
  solver::PseudoTransientSolver{2,T,BE}, mesh, Δt, atol=eps(T), rtol=sqrt(eps(T))
) where {T,BE<:CPU}

  #
  domain = solver.iterators.domain.cartesian

  u = solver.u
  u_prev = solver.u_prev

  qξ, qη = solver.q
  dτ_ρ = solver.dτ_ρ
  source_term = solver.source_term

  ξ_x = mesh.cell_center_metrics.ξ.x₁
  ξ_y = mesh.cell_center_metrics.ξ.x₂
  η_x = mesh.cell_center_metrics.η.x₁
  η_y = mesh.cell_center_metrics.η.x₂

  @batch for idx in domain
    i, j = idx.I

    ξx = ξ_x[i, j]
    ξy = ξ_y[i, j]
    ηx = η_x[i, j]
    ηy = η_y[i, j]

    _dqξ = qξ[i, j] - qξ[i - 1, j]
    _dqη = qη[i, j] - qη[i, j - 1]

    # _dqξ = _dqξ * (abs(_dqξ) >= atol)
    # _dqη = _dqη * (abs(_dqη) >= atol)

    _dqξ = _dqξ * (abs(qξ[i, j] * rtol) < abs(_dqξ))
    _dqη = _dqη * (abs(qη[i, j] * rtol) < abs(_dqη))

    ∂qξ∂ξ = (ξx^2 + ξy^2) * _dqξ
    ∂qη∂η = (ηx^2 + ηy^2) * _dqη

    ∇q = ∂qξ∂ξ + ∂qη∂η

    u[idx] = (
      (u[idx] + dτ_ρ[idx] * (u_prev[idx] / Δt - ∇q + source_term[idx])) /
      (1 + dτ_ρ[idx] / Δt)
    )
  end

  return nothing
end

# ------------------------------------------------------------------------------------------
# 3D
# ------------------------------------------------------------------------------------------
function update_orthogonal!(
  solver::PseudoTransientSolver{3,T,BE}, mesh, Δt, ϵ=5eps(T)
) where {T,BE<:CPU}

  #
  domain = solver.iterators.domain.cartesian

  u = solver.u
  u_prev = solver.u_prev

  qξ, qη, qζ = solver.q
  dτ_ρ = solver.dτ_ρ
  source_term = solver.source_term

  ξx = mesh.cell_center_metrics.ξ.x₁
  ξy = mesh.cell_center_metrics.ξ.x₂
  ξz = mesh.cell_center_metrics.ξ.x₃

  ηx = mesh.cell_center_metrics.η.x₁
  ηy = mesh.cell_center_metrics.η.x₂
  ηz = mesh.cell_center_metrics.η.x₃

  ζx = mesh.cell_center_metrics.ζ.x₁
  ζy = mesh.cell_center_metrics.ζ.x₂
  ζz = mesh.cell_center_metrics.ζ.x₃

  @batch for idx in domain
    i, j, k = idx.I

    _dqξ = qξ[i, j, k] - qξ[i - 1, j, k]
    _dqη = qη[i, j, k] - qη[i, j - 1, k]
    _dqζ = qζ[i, j, k] - qζ[i, j, k - 1]

    _dqξ = _dqξ * (abs(_dqξ) >= ϵ)
    _dqη = _dqη * (abs(_dqη) >= ϵ)
    _dqζ = _dqζ * (abs(_dqζ) >= ϵ)

    ∂qξ∂ξ = (ξx[idx]^2 + ξy[idx]^2 + ξz[idx]^2) * _dqξ
    ∂qη∂η = (ηx[idx]^2 + ηy[idx]^2 + ηz[idx]^2) * _dqη
    ∂qζ∂ζ = (ζx[idx]^2 + ζy[idx]^2 + ζz[idx]^2) * _dqζ

    ∇q = ∂qξ∂ξ + ∂qη∂η + ∂qζ∂ζ

    u[idx] = (
      (u[idx] + dτ_ρ[idx] * (u_prev[idx] / Δt - ∇q + source_term[idx])) /
      (1 + dτ_ρ[idx] / Δt)
    )
  end

  return nothing
end