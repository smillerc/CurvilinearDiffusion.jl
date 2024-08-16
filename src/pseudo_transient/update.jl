# 2D update
function _update2d_orthog_mesh2!(
  u, u_prev, ξx, ξy, ηx, ηy, qξ_ᵢⱼ, qξ_ᵢ₋₁ⱼ, qη_ᵢⱼ, qη_ᵢⱼ₋₁, dτ_ρ, source_term, dt
)
  ∂qξ∂ξ = (ξx^2 + ξy^2) * (qξ_ᵢⱼ - qξ_ᵢ₋₁ⱼ)
  ∂qη∂η = (ηx^2 + ηy^2) * (qη_ᵢⱼ - qη_ᵢⱼ₋₁)

  ∇q = ∂qξ∂ξ + ∂qη∂η

  unew = (u + dτ_ρ * (u_prev / dt - ∇q + source_term)) / (1 + dτ_ρ / dt)
  return unew
end

@kernel inbounds = true function _update_kernel!(
  u::AbstractArray{T,2},
  @Const(u_prev),
  # @Const(J_i₊½),
  # @Const(J_j₊½),
  @Const(ξ_x),
  @Const(ξ_y),
  @Const(η_x),
  @Const(η_y),
  # @Const(ξ̂x_ᵢ₊½),
  # @Const(ξ̂y_ᵢ₊½),
  # @Const(η̂x_ᵢ₊½),
  # @Const(η̂y_ᵢ₊½),
  # @Const(ξ̂x_ⱼ₊½),
  # @Const(ξ̂y_ⱼ₊½),
  # @Const(η̂x_ⱼ₊½),
  # @Const(η̂y_ⱼ₊½),
  @Const(qξ),
  @Const(qη),
  @Const(dτ_ρ),
  @Const(source_term),
  @Const(dt),
  @Const(I0),
) where {T}

  #
  idx = @index(Global, Cartesian)
  idx += I0

  i, j = idx.I

  # Jᵢ₊½⁻¹ = inv(J_i₊½[i, j])
  # Jⱼ₊½⁻¹ = inv(J_j₊½[i, j])
  # Jᵢ₋½⁻¹ = inv(J_i₊½[i - 1, j])
  # Jⱼ₋½⁻¹ = inv(J_j₊½[i, j - 1])

  ξx = ξ_x[i, j]
  ξy = ξ_y[i, j]
  ηx = η_x[i, j]
  ηy = η_y[i, j]

  # ξxᵢ₊½ = ξ̂x_ᵢ₊½[i, j] * Jᵢ₊½⁻¹
  # ξyᵢ₊½ = ξ̂y_ᵢ₊½[i, j] * Jᵢ₊½⁻¹
  # ηxᵢ₊½ = η̂x_ᵢ₊½[i, j] * Jᵢ₊½⁻¹
  # ηyᵢ₊½ = η̂y_ᵢ₊½[i, j] * Jᵢ₊½⁻¹

  # ξxᵢ₋½ = ξ̂x_ᵢ₊½[i - 1, j] * Jᵢ₋½⁻¹
  # ξyᵢ₋½ = ξ̂y_ᵢ₊½[i - 1, j] * Jᵢ₋½⁻¹
  # ηxᵢ₋½ = η̂x_ᵢ₊½[i - 1, j] * Jᵢ₋½⁻¹
  # ηyᵢ₋½ = η̂y_ᵢ₊½[i - 1, j] * Jᵢ₋½⁻¹

  # ξxⱼ₊½ = ξ̂x_ⱼ₊½[i, j] * Jⱼ₊½⁻¹
  # ξyⱼ₊½ = ξ̂y_ⱼ₊½[i, j] * Jⱼ₊½⁻¹
  # ηxⱼ₊½ = η̂x_ⱼ₊½[i, j] * Jⱼ₊½⁻¹
  # ηyⱼ₊½ = η̂y_ⱼ₊½[i, j] * Jⱼ₊½⁻¹

  # ξxⱼ₋½ = ξ̂x_ⱼ₊½[i, j - 1] * Jⱼ₋½⁻¹
  # ξyⱼ₋½ = ξ̂y_ⱼ₊½[i, j - 1] * Jⱼ₋½⁻¹
  # ηxⱼ₋½ = η̂x_ⱼ₊½[i, j - 1] * Jⱼ₋½⁻¹
  # ηyⱼ₋½ = η̂y_ⱼ₊½[i, j - 1] * Jⱼ₋½⁻¹

  # flux divergence

  # aᵢⱼ = (
  #   ξx * (ξxᵢ₊½ - ξxᵢ₋½) +
  #   ξy * (ξyᵢ₊½ - ξyᵢ₋½) +
  #   ηx * (ξxⱼ₊½ - ξxⱼ₋½) +
  #   ηy * (ξyⱼ₊½ - ξyⱼ₋½)
  # )

  # bᵢⱼ = (
  #   ξx * (ηxᵢ₊½ - ηxᵢ₋½) +
  #   ξy * (ηyᵢ₊½ - ηyᵢ₋½) +
  #   ηx * (ηxⱼ₊½ - ηxⱼ₋½) +
  #   ηy * (ηyⱼ₊½ - ηyⱼ₋½)
  # )

  ∂qξ∂ξ = (ξx^2 + ξy^2) * (qξ[i, j] - qξ[i - 1, j])
  ∂qη∂η = (ηx^2 + ηy^2) * (qη[i, j] - qη[i, j - 1])

  # ∂qξ∂η =
  #   0.25(ηx * ξx + ηy * ξy) * (
  #     (qξ[i, j + 1] + qξ[i - 1, j + 1]) - # take average on either side
  #     (qξ[i, j - 1] + qξ[i - 1, j - 1])   # and do diff in j
  #   )

  # ∂qη∂ξ =
  #   0.25(ηx * ξx + ηy * ξy) * (
  #     (qη[i + 1, j] + qη[i + 1, j - 1]) - # take average on either side
  #     (qη[i - 1, j] + qη[i - 1, j - 1])   # and do diff in i
  #   )

  # ∂H∂ξ = aᵢⱼ * 0.5(qξ[i, j] + qξ[i - 1, j]) # ∂u/∂ξ + non-orth terms
  # ∂H∂η = bᵢⱼ * 0.5(qη[i, j] + qη[i, j - 1]) # ∂u/∂η + non-orth terms

  ∇q = ∂qξ∂ξ + ∂qη∂η # + ∂qξ∂η + ∂qη∂ξ + ∂H∂ξ + ∂H∂η

  u[idx] = (
    (u[idx] + dτ_ρ[idx] * (u_prev[idx] / dt - ∇q + source_term[idx])) / (1 + dτ_ρ[idx] / dt)
  )

  #
end

@kernel function _update_kernel2!(
  u,
  @Const(u_prev),
  cell_center_metrics, # applying @Const to a struct array causes problems
  edge_metrics, # applying @Const to a struct array causes problems
  @Const(flux),
  @Const(dτ_ρ),
  @Const(source_term),
  @Const(dt),
  @Const(I0),
)
  idx = @index(Global, Cartesian)
  idx += I0

  @inbounds begin
    @inline ∇q = flux_divergence(flux, cell_center_metrics, edge_metrics, idx)

    u[idx] = (
      (u[idx] + dτ_ρ[idx] * (u_prev[idx] / dt - ∇q + source_term[idx])) /
      (1 + dτ_ρ[idx] / dt)
    )
  end
end

NVTX.@annotate function compute_update!(
  solver::PseudoTransientSolver{N,T,BE}, mesh, Δt
) where {N,T,BE<:GPU}
  domain = solver.iterators.domain.cartesian

  iaxis, jaxis = (1, 2)
  ᵢ₋₁ⱼ_domain = shift(domain, iaxis, -1)
  ᵢⱼ₋₁_domain = shift(domain, jaxis, -1)

  uᵢⱼ = @view solver.u[domain]
  u_prevᵢⱼ = @view solver.u_prev[domain]
  qξ_ᵢⱼ = @view solver.q.x[domain]
  qη_ᵢⱼ = @view solver.q.y[domain]
  qξ_ᵢ₋₁ⱼ = @view solver.q.x[ᵢ₋₁ⱼ_domain]
  qη_ᵢⱼ₋₁ = @view solver.q.y[ᵢⱼ₋₁_domain]

  ξx = @view mesh.cell_center_metrics.ξ.x₁[domain]
  ξy = @view mesh.cell_center_metrics.ξ.x₂[domain]
  ηx = @view mesh.cell_center_metrics.η.x₁[domain]
  ηy = @view mesh.cell_center_metrics.η.x₂[domain]

  dτ_ρ = @view solver.dτ_ρ[domain]
  source_term = @view solver.source_term[domain]

  @. uᵢⱼ = _update2d_orthog_mesh2!(
    uᵢⱼ, u_prevᵢⱼ, ξx, ξy, ηx, ηy, qξ_ᵢⱼ, qξ_ᵢ₋₁ⱼ, qη_ᵢⱼ, qη_ᵢⱼ₋₁, dτ_ρ, source_term, Δt
  )

  # _update2d_orthog_mesh!.(
  #   uᵢⱼ, u_prevᵢⱼ, ξx, ξy, ηx, ηy, qξ_ᵢⱼ, qξ_ᵢ₋₁ⱼ, qη_ᵢⱼ, qη_ᵢⱼ₋₁, dτ_ρ, source_term, Δt
  # )

  return nothing
end

function compute_update456!(
  solver::PseudoTransientSolver{N,T,BE}, mesh, Δt
) where {N,T,BE<:GPU}
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _update_kernel2!(solver.backend)(
    solver.u,
    solver.u_prev,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    solver.q,
    solver.dτ_ρ,
    solver.source_term,
    Δt,
    idx_offset;
    ndrange=size(domain),
  )

  KernelAbstractions.synchronize(solver.backend)
  return nothing
end

# function compute_update!(
#   solver::PseudoTransientSolver{N,T,BE}, mesh, Δt
# ) where {N,T,BE<:GPU}
#   domain = solver.iterators.domain.cartesian
#   idx_offset = first(domain) - oneunit(first(domain))

#   _update_kernel!(solver.backend)(
#     solver.u,
#     solver.u_prev,
#     # mesh.edge_metrics.i₊½.J,
#     # mesh.edge_metrics.j₊½.J,
#     mesh.cell_center_metrics.ξ.x₁,
#     mesh.cell_center_metrics.ξ.x₂,
#     mesh.cell_center_metrics.η.x₁,
#     mesh.cell_center_metrics.η.x₂,
#     # mesh.edge_metrics.i₊½.ξ̂.x₁,
#     # mesh.edge_metrics.i₊½.ξ̂.x₂,
#     # mesh.edge_metrics.i₊½.η̂.x₁,
#     # mesh.edge_metrics.i₊½.η̂.x₂,
#     # mesh.edge_metrics.j₊½.ξ̂.x₁,
#     # mesh.edge_metrics.j₊½.ξ̂.x₂,
#     # mesh.edge_metrics.j₊½.η̂.x₁,
#     # mesh.edge_metrics.j₊½.η̂.x₂,
#     solver.q.x,
#     solver.q.y,
#     solver.dτ_ρ,
#     solver.source_term,
#     Δt,
#     idx_offset;
#     ndrange=size(domain),
#   )

#   KernelAbstractions.synchronize(solver.backend)
#   return nothing
# end

# function compute_update!(
#   solver::PseudoTransientSolver{N,T,BE}, mesh, Δt
# ) where {T,N,BE<:CPU}

#   #
#   domain = solver.iterators.domain.cartesian

#   u = solver.u
#   u_prev = solver.u_prev
#   cell_center_metrics = mesh.cell_center_metrics
#   edge_metrics = mesh.edge_metrics

#   flux = solver.q
#   dτ_ρ = solver.dτ_ρ
#   source_term = solver.source_term

#   @batch for idx in domain
#     @inline ∇q = flux_divergence(flux, cell_center_metrics, edge_metrics, idx)

#     u[idx] = (
#       (u[idx] + dτ_ρ[idx] * (u_prev[idx] / Δt - ∇q + source_term[idx])) /
#       (1 + dτ_ρ[idx] / Δt)
#     )
#   end

#   return nothing
# end

function compute_update!(
  solver::PseudoTransientSolver{N,T,BE}, mesh, Δt
) where {T,N,BE<:CPU}

  #
  domain = solver.iterators.domain.cartesian

  u = solver.u
  u_prev = solver.u_prev
  cell_center_metrics = mesh.cell_center_metrics
  edge_metrics = mesh.edge_metrics

  qξ, qη = solver.q
  dτ_ρ = solver.dτ_ρ
  source_term = solver.source_term

  ξ_x = cell_center_metrics.ξ.x₁ #[i, j]
  ξ_y = cell_center_metrics.ξ.x₂ #[i, j]
  η_x = cell_center_metrics.η.x₁ #[i, j]
  η_y = cell_center_metrics.η.x₂ #[i, j]

  @batch for idx in domain
    i, j = idx.I

    Jᵢ₊½ = inv(edge_metrics.i₊½.J[i, j])
    Jⱼ₊½ = inv(edge_metrics.j₊½.J[i, j])
    Jᵢ₋½ = inv(edge_metrics.i₊½.J[i - 1, j])
    Jⱼ₋½ = inv(edge_metrics.j₊½.J[i, j - 1])

    ξx = ξ_x[i, j] #cell_center_metrics.ξ.x₁[i, j]
    ξy = ξ_y[i, j] #cell_center_metrics.ξ.x₂[i, j]
    ηx = η_x[i, j] #cell_center_metrics.η.x₁[i, j]
    ηy = η_y[i, j] #cell_center_metrics.η.x₂[i, j]

    ξxᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[i, j] * Jᵢ₊½
    ξyᵢ₊½ = edge_metrics.i₊½.ξ̂.x₂[i, j] * Jᵢ₊½
    ηxᵢ₊½ = edge_metrics.i₊½.η̂.x₁[i, j] * Jᵢ₊½
    ηyᵢ₊½ = edge_metrics.i₊½.η̂.x₂[i, j] * Jᵢ₊½

    ξxᵢ₋½ = edge_metrics.i₊½.ξ̂.x₁[i - 1, j] * Jᵢ₋½
    ξyᵢ₋½ = edge_metrics.i₊½.ξ̂.x₂[i - 1, j] * Jᵢ₋½
    ηxᵢ₋½ = edge_metrics.i₊½.η̂.x₁[i - 1, j] * Jᵢ₋½
    ηyᵢ₋½ = edge_metrics.i₊½.η̂.x₂[i - 1, j] * Jᵢ₋½

    ξxⱼ₊½ = edge_metrics.j₊½.ξ̂.x₁[i, j] * Jⱼ₊½
    ξyⱼ₊½ = edge_metrics.j₊½.ξ̂.x₂[i, j] * Jⱼ₊½
    ηxⱼ₊½ = edge_metrics.j₊½.η̂.x₁[i, j] * Jⱼ₊½
    ηyⱼ₊½ = edge_metrics.j₊½.η̂.x₂[i, j] * Jⱼ₊½

    ξxⱼ₋½ = edge_metrics.j₊½.ξ̂.x₁[i, j - 1] * Jⱼ₋½
    ξyⱼ₋½ = edge_metrics.j₊½.ξ̂.x₂[i, j - 1] * Jⱼ₋½
    ηxⱼ₋½ = edge_metrics.j₊½.η̂.x₁[i, j - 1] * Jⱼ₋½
    ηyⱼ₋½ = edge_metrics.j₊½.η̂.x₂[i, j - 1] * Jⱼ₋½

    # flux divergence

    aᵢⱼ = (
      ξx * (ξxᵢ₊½ - ξxᵢ₋½) +
      ξy * (ξyᵢ₊½ - ξyᵢ₋½) +
      ηx * (ξxⱼ₊½ - ξxⱼ₋½) +
      ηy * (ξyⱼ₊½ - ξyⱼ₋½)
    )

    bᵢⱼ = (
      ξx * (ηxᵢ₊½ - ηxᵢ₋½) +
      ξy * (ηyᵢ₊½ - ηyᵢ₋½) +
      ηx * (ηxⱼ₊½ - ηxⱼ₋½) +
      ηy * (ηyⱼ₊½ - ηyⱼ₋½)
    )

    ∂qξ∂ξ = (ξx^2 + ξy^2) * (qξ[i, j] - qξ[i - 1, j])
    ∂qη∂η = (ηx^2 + ηy^2) * (qη[i, j] - qη[i, j - 1])

    ∂qξ∂η =
      0.25(ηx * ξx + ηy * ξy) * (
        (qξ[i, j + 1] + qξ[i - 1, j + 1]) - # take average on either side
        (qξ[i, j - 1] + qξ[i - 1, j - 1])   # and do diff in j
      )

    ∂qη∂ξ =
      0.25(ηx * ξx + ηy * ξy) * (
        (qη[i + 1, j] + qη[i + 1, j - 1]) - # take average on either side
        (qη[i - 1, j] + qη[i - 1, j - 1])   # and do diff in i
      )

    ∂H∂ξ = aᵢⱼ * 0.5(qξ[i, j] + qξ[i - 1, j]) # ∂u/∂ξ + non-orth terms
    ∂H∂η = bᵢⱼ * 0.5(qη[i, j] + qη[i, j - 1]) # ∂u/∂η + non-orth terms

    ∇q = ∂qξ∂ξ + ∂qη∂η + ∂qξ∂η + ∂qη∂ξ + ∂H∂ξ + ∂H∂η

    u[idx] = (
      (u[idx] + dτ_ρ[idx] * (u_prev[idx] / Δt - ∇q + source_term[idx])) /
      (1 + dτ_ρ[idx] / Δt)
    )
  end

  return nothing
end