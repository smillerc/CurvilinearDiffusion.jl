#
@kernel function _update_kernel!(
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

function compute_update!(
  solver::PseudoTransientSolver{N,T,BE}, mesh, Δt
) where {T,N,BE<:CPU}

  #
  domain = solver.iterators.domain.cartesian

  u = solver.u
  u_prev = solver.u_prev
  cell_center_metrics = mesh.cell_center_metrics
  edge_metrics = mesh.edge_metrics

  flux = solver.q
  dτ_ρ = solver.dτ_ρ
  source_term = solver.source_term

  @batch for idx in domain
    @inline ∇q = flux_divergence(flux, cell_center_metrics, edge_metrics, idx, mesh)

    u[idx] = (
      (u[idx] + dτ_ρ[idx] * (u_prev[idx] / Δt - ∇q + source_term[idx])) /
      (1 + dτ_ρ[idx] / Δt)
    )
  end

  return nothing
end
