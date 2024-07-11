
@kernel function _update_resid_arb_mesh!(
  resid, cell_center_metrics, edge_metrics, α, u, u_prev, q, source_term, dt, index_offset
)
  idx = @index(Global, Cartesian)
  idx += index_offset

  @inbounds begin
    @inline ∇q = _arbitrary_flux_divergence(q, u, α, cell_center_metrics, edge_metrics, idx)

    resid[idx] = -(u[idx] - u_prev[idx]) / dt - ∇q + source_term[idx]
  end
end

@kernel function _update_resid_orth_mesh!(
  resid, cell_center_metrics, α, u, u_prev, q, source_term, dt, index_offset
)
  idx = @index(Global, Cartesian)
  idx += index_offset

  @inbounds begin
    @inline ∇q = _orthogonal_flux_divergence(q, u, α, cell_center_metrics, idx)

    resid[idx] = -(u[idx] - u_prev[idx]) / dt - ∇q + source_term[idx]
  end
end

"""
    update_residual!(solver::PseudoTransientSolver, mesh, Δt)

"""
function update_residual!(solver::PseudoTransientSolver, mesh, Δt)
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  # if mesh.is_orthogonal
  #   _update_resid_orth_mesh!(solver.backend)(
  #     solver.residual,
  #     mesh.cell_center_metrics,
  #     solver.α,
  #     solver.u,
  #     solver.u_prev,
  #     solver.q′,
  #     solver.source_term,
  #     Δt,
  #     idx_offset;
  #     ndrange=size(domain),
  #   )
  # else
  _update_resid_arb_mesh!(solver.backend)(
    solver.residual,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    solver.α,
    solver.u,
    solver.u_prev,
    solver.q′,
    solver.source_term,
    Δt,
    idx_offset;
    ndrange=size(domain),
  )
  # end

  KernelAbstractions.synchronize(solver.backend)
  return nothing
end
