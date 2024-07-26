function L2_norm(A, ::GPU)
  _norm = sqrt(mapreduce(x -> (x^2), +, A) / length(A))
  return _norm
end

function L2_norm(A, ::CPU)
  _L2_norm(A, Val(nthreads()))
end

function _L2_norm(a, ::Val{nchunks}) where {nchunks}
  _numer = @MVector zeros(nchunks)

  @batch for idx in eachindex(a)
    ichunk = threadid()

    _numer[ichunk] += a[idx]^2
  end

  return sqrt(sum(_numer) / length(a))
end

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

"""
    update_residual!(solver::PseudoTransientSolver, mesh, Δt)

"""
function update_residual!(
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

NVTX.@annotate function update_residual!(
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