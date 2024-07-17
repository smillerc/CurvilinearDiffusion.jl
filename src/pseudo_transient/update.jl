#
@kernel function _update_kernel!(
  u::AbstractArray{T,2},
  u_prev,
  cell_center_metrics, # applying @Const to a struct array causes problems
  edge_metrics, # applying @Const to a struct array causes problems
  α,
  q,
  dτ_ρ,
  source_term,
  dt,
  I0,
) where {T}
  idx = @index(Global, Cartesian)
  idx += I0

  qᵢ = q.x
  qⱼ = q.y

  @inbounds begin
    ∇q = flux_divergence((qᵢ, qⱼ), cell_center_metrics, edge_metrics, idx)

    u[idx] = (
      (u[idx] + dτ_ρ[idx] * (u_prev[idx] / dt - ∇q + source_term[idx])) /
      (1 + dτ_ρ[idx] / dt)
    )
  end
end

function compute_update!(
  solver::PseudoTransientSolver{N,T,BE}, mesh, Δt
) where {N,T,BE<:GPU}
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _update_kernel!(solver.backend)(
    solver.u,
    solver.u_prev,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    solver.α,
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

function compute_update!(solver::PseudoTransientSolver{2,T,BE}, mesh, Δt) where {T,BE<:CPU}

  #
  domain = solver.iterators.domain.cartesian

  u = solver.u
  u_prev = solver.u_prev
  cell_center_metrics = mesh.cell_center_metrics
  edge_metrics = mesh.edge_metrics

  qᵢ, qⱼ = solver.q
  dτ_ρ = solver.dτ_ρ
  source_term = solver.source_term

  @batch for idx in domain
    @inline ∇q = flux_divergence((qᵢ, qⱼ), cell_center_metrics, edge_metrics, idx)

    u[idx] = (
      (u[idx] + dτ_ρ[idx] * (u_prev[idx] / Δt - ∇q + source_term[idx])) /
      (1 + dτ_ρ[idx] / Δt)
    )
  end

  return nothing
end