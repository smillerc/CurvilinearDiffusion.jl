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

@kernel inbounds = true function _update_resid_kernel!(
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

  @inline ∇q = flux_divergence(flux, cell_center_metrics, edge_metrics, idx)

  residuals[idx] = -(u[idx] - u_prev[idx]) / dt - ∇q + source_term[idx]
end

"""
    update_residual!(solver::PseudoTransientSolver, mesh, Δt)

"""
function update_residual!(solver::PseudoTransientSolver{N,T}, mesh, Δt) where {N,T}
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _update_resid_kernel!(solver.backend)(
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
