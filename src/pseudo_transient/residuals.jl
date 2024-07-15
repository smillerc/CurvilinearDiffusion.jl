
@kernel function _update_resid_arb_mesh!(
  resid,
  cell_center_metrics,
  edge_metrics,
  α,
  u,
  u_prev,
  q,
  source_term,
  smax,
  dt,
  index_offset,
)
  idx = @index(Global, Cartesian)
  idx += index_offset

  @inbounds begin
    @inline ∇q = _arbitrary_flux_divergence(q, u, α, cell_center_metrics, edge_metrics, idx)

    resid[idx] = abs(-(u[idx] - u_prev[idx]) / dt - ∇q + source_term[idx])#/ smax
    # resid[idx] /= u[idx]
    # resid[idx] = -(u[idx] - u_prev[idx]) / dt - ∇q
  end
end

@kernel function _update_resid_orth_mesh!(
  resid, cell_center_metrics, α, u, u_prev, q, source_term, dt, index_offset
)
  idx = @index(Global, Cartesian)
  idx += index_offset

  @inbounds begin
    @inline ∇q = _orthogonal_flux_divergence(q, u, α, cell_center_metrics, idx)

    resid[idx] = abs(-(u[idx] - u_prev[idx]) / dt - ∇q + source_term[idx])
  end
end

@kernel function _resid_update_all!(resid, q, q′, index_offset)
  idx = @index(Global, Cartesian)
  idx += index_offset

  @inbounds begin
    resid_x = abs(q.x[idx] - q′.x[idx])
    resid_x = resid_x * (abs(resid_x) >= 1e-16)
    # norm_q = abs(q′.x[idx])
    # norm_q = max(abs(q.x[idx]) + abs(q′.x[idx]))
    # x_denom = ifelse(isfinite(norm_q) && !iszero(norm_q), inv(norm_q), 0.0)
    resid[idx] = resid_x # * x_denom
  end
end

"""
    update_residual!(solver::PseudoTransientSolver, mesh, Δt)

"""
function update_residual!(solver::PseudoTransientSolver, mesh, Δt)
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _resid_update_all!(solver.backend)(
    solver.residual, solver.q, solver.q′, idx_offset; ndrange=size(domain)
  )

  KernelAbstractions.synchronize(solver.backend)
  return nothing
end
