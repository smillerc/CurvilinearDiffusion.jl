function update_nonorthogonal!(
  solver::PseudoTransientSolver{1,T,BE}, mesh, Δt
) where {T,BE<:CPU}
  update_orthogonal!(solver, mesh, Δt) # 1D is always orthogonal
end

function update_nonorthogonal!(
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
    @inline ∇q = flux_divergence(flux, cell_center_metrics, edge_metrics, idx)

    u[idx] = (
      (u[idx] + dτ_ρ[idx] * (u_prev[idx] / Δt - ∇q + source_term[idx])) /
      (1 + dτ_ρ[idx] / Δt)
    )
  end

  return nothing
end