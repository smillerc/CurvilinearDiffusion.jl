@kernel function _update_resid!(
  residuals,
  cell_center_metrics,
  edge_metrics,
  @Const(u),
  @Const(u_prev),
  @Const(flux),
  @Const(source_term),
  @Const(dt),
  @Const(I0),
)
  idx = @index(Global, Cartesian)
  idx += I0

  @inbounds begin
    @inline ∇q = flux_divergence(flux, cell_center_metrics, edge_metrics, idx)

    residuals[idx] = -(u[idx] - u_prev[idx]) / dt - ∇q + source_term[idx]
  end
end

function update_residuals_nonorthogonal!(
  solver::PseudoTransientSolver{N,T,BE}, mesh, Δt
) where {N,T,BE<:GPU}

  #
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _update_resid!(solver.backend)(
    solver.res,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    solver.u,
    solver.u_prev,
    solver.q′,
    solver.source_term,
    Δt,
    idx_offset;
    ndrange=size(domain),
  )

  KernelAbstractions.synchronize(solver.backend)
  return nothing
end