function update_residuals_nonorthogonal!(
  solver::PseudoTransientSolver{N,T,BE}, mesh, Δt
) where {N,T,BE<:CPU}

  #
  domain = solver.iterators.domain.cartesian
  u = solver.u
  u_prev = solver.u_prev
  flux = solver.q′
  cell_center_metrics = mesh.cell_center_metrics
  edge_metrics = mesh.edge_metrics
  residuals = solver.res
  source_term = solver.source_term

  @batch for idx in domain
    @inline ∇q = flux_divergence(flux, cell_center_metrics, edge_metrics, idx)

    residuals[idx] = -(u[idx] - u_prev[idx]) / Δt - ∇q + source_term[idx]
  end

  return nothing
end