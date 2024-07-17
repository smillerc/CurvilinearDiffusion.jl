function L2_norm(A)
  _norm = sqrt(mapreduce(x -> x^2, +, A)) / sqrt(length(A))
  return _norm
end

@kernel function _update_resid!(
  residuals::AbstractArray{T,2},
  cell_center_metrics,
  edge_metrics,
  α,
  u,
  u_prev,
  flux,
  source_term,
  dt,
  I0,
) where {T}
  idx = @index(Global, Cartesian)
  idx += I0

  qᵢ = flux.x
  qⱼ = flux.y

  @inbounds begin
    @inline ∇q = flux_divergence((qᵢ, qⱼ), cell_center_metrics, edge_metrics, idx)

    residuals[idx] = -(u[idx] - u_prev[idx]) / dt - ∇q + source_term[idx]
  end
end

# """
#     update_residual!(solver::PseudoTransientSolver, mesh, Δt)

# """
# function update_residual!(solver::PseudoTransientSolver, mesh, Δt)
#   domain = solver.iterators.domain.cartesian
#   idx_offset = first(domain) - oneunit(first(domain))

#   _update_resid!(solver.backend)(
#     solver.res,
#     mesh.cell_center_metrics,
#     mesh.edge_metrics,
#     solver.α,
#     solver.u,
#     solver.u_prev,
#     solver.q′,
#     solver.source_term,
#     Δt,
#     idx_offset;
#     ndrange=size(domain),
#   )

#   KernelAbstractions.synchronize(solver.backend)
#   return nothing
# end

function update_residual!(
  solver::PseudoTransientSolver{N,T,BE}, mesh, Δt
) where {N,T,BE<:CPU}

  #
  domain = solver.iterators.domain.cartesian
  # idx_offset = first(domain) - oneunit(first(domain))

  # _update_resid!(solver.backend)(
  #   solver.res,
  #   mesh.cell_center_metrics,
  #   mesh.edge_metrics,
  #   solver.α,
  #   solver.u,
  #   solver.u_prev,
  #   solver.q′,
  #   solver.source_term,
  #   Δt,
  #   idx_offset;
  #   ndrange=size(domain),
  # )

  u = solver.u
  u_prev = solver.u_prev
  qᵢ, qⱼ = solver.q′
  cell_center_metrics = mesh.cell_center_metrics
  edge_metrics = mesh.edge_metrics
  residuals = solver.res
  source_term = solver.source_term

  @batch for idx in domain
    @inline ∇q = flux_divergence((qᵢ, qⱼ), cell_center_metrics, edge_metrics, idx)

    residuals[idx] = -(u[idx] - u_prev[idx]) / Δt - ∇q + source_term[idx]
  end

  return nothing
end