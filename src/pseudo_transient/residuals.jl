function L2_norm(A)
  _norm = sqrt(mapreduce(x -> x^2, +, A)) / sqrt(length(A))
  return _norm
end

@kernel function _update_resid!(
  resid::AbstractArray{T,2},
  cell_center_metrics,
  edge_metrics,
  α,
  H,
  H_prev,
  flux,
  source_term,
  dt,
  I0,
) where {T}
  idx = @index(Global, Cartesian)
  idx += I0

  qᵢ = flux.x
  qⱼ = flux.y

  ∇q = flux_divergence((qᵢ, qⱼ), H, α, cell_center_metrics, edge_metrics, idx)

  resid[idx] = -(H[idx] - H_prev[idx]) / dt - ∇q + source_term[idx]
end

"""
    update_residual!(solver::PseudoTransientSolver, mesh, Δt)

"""
function update_residual!(solver::PseudoTransientSolver, mesh, Δt)
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _update_resid!(solver.backend)(
    solver.res,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    solver.α,
    solver.H,
    solver.H_prev,
    solver.qH_2,
    solver.source_term,
    Δt,
    idx_offset;
    ndrange=size(domain),
  )

  KernelAbstractions.synchronize(solver.backend)
  return nothing
end