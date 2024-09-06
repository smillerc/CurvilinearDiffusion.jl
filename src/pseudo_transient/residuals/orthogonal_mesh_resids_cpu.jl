
# ------------------------------------------------------------------------------------------
# 1D
# ------------------------------------------------------------------------------------------

function update_residuals_orthogonal!(
  solver::PseudoTransientSolver{1,T,BE}, mesh, Δt, ϵ=eps(T)
) where {T,BE<:CPU}

  #
  domain = solver.iterators.domain.cartesian
  residuals = solver.res

  u = solver.u
  u_prev = solver.u_prev

  qξ = solver.q′.x
  source_term = solver.source_term

  ξx = mesh.cell_center_metrics.ξ.x₁

  @batch for idx in domain
    i, = idx.I

    qξᵢ₊½ = qξ[i]
    qξᵢ₋½ = qξ[i - 1]

    @inline ∇q = flux_divergence_orth(qξᵢ₊½, qξᵢ₋½, ξx[idx])

    uⁿ = u[idx]
    uⁿ⁻¹ = u_prev[idx]
    du = uⁿ - uⁿ⁻¹
    du = du * !isapprox(uⁿ, uⁿ⁻¹; rtol=ϵ)

    residuals[idx] = -du / Δt - ∇q + source_term[idx]
  end

  return nothing
end

# ------------------------------------------------------------------------------------------
# 2D
# ------------------------------------------------------------------------------------------
function update_residuals_orthogonal!(
  solver::PseudoTransientSolver{2,T,BE}, mesh, Δt, ϵ=eps(T)
) where {T,BE<:CPU}

  #
  domain = solver.iterators.domain.cartesian

  u = solver.u
  u_prev = solver.u_prev
  residuals = solver.res

  qξ, qη = solver.q′
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

    qξᵢ₊½ = qξ[i, j]
    qξᵢ₋½ = qξ[i - 1, j]
    qηⱼ₊½ = qη[i, j]
    qηⱼ₋½ = qη[i, j - 1]

    @inline ∇q = flux_divergence_orth(qξᵢ₊½, qξᵢ₋½, qηⱼ₊½, qηⱼ₋½, ξx, ξy, ηx, ηy)

    uⁿ = u[idx]
    uⁿ⁻¹ = u_prev[idx]
    du = uⁿ - uⁿ⁻¹
    du = du * !isapprox(uⁿ, uⁿ⁻¹; rtol=ϵ)

    residuals[idx] = -du / Δt - ∇q + source_term[idx]
  end

  return nothing
end

# ------------------------------------------------------------------------------------------
# 3D
# ------------------------------------------------------------------------------------------
function update_residuals_orthogonal!(
  solver::PseudoTransientSolver{3,T,BE}, mesh, Δt, ϵ=eps(T)
) where {T,BE<:CPU}

  #
  domain = solver.iterators.domain.cartesian

  u = solver.u
  u_prev = solver.u_prev
  residuals = solver.res

  qξ, qη, qζ = solver.q′
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

    qξᵢ₊½ = qξ[i, j, k]
    qξᵢ₋½ = qξ[i - 1, j, k]
    qηⱼ₊½ = qη[i, j, k]
    qηⱼ₋½ = qη[i, j - 1, k]
    qζₖ₊½ = qζ[i, j, k]
    qζₖ₋½ = qζ[i, j, k - 1]

    @inline ∇q = flux_divergence_orth(
      qξᵢ₊½, qξᵢ₋½, qηⱼ₊½, qηⱼ₋½, qζₖ₊½, qζₖ₋½, ξx, ξy, ξz, ηx, ηy, ηz, ζx, ζy, ζz
    )

    uⁿ = u[idx]
    uⁿ⁻¹ = u_prev[idx]
    du = uⁿ - uⁿ⁻¹
    du = du * !isapprox(uⁿ, uⁿ⁻¹; rtol=ϵ) # epsilon check

    residuals[idx] = -du / Δt - ∇q + source_term[idx]
  end

  return nothing
end