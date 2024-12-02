function L2_norm(A, ::GPU)
  # _norm = sqrt(mapreduce(x -> (x^2), +, A) / length(A))
  # return _norm
  # return norm(A) / sqrt(length(A))
  return norm(A, Inf) / sqrt(length(A))
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
  cache,
  cell_center_metrics,
  @Const(u),
  @Const(u_prev),
  @Const(flux),
  @Const(source_term),
  @Const(dt),
  @Const(ϵ),
  @Const(I0),
)
  idx = @index(Global, Cartesian)
  idx += I0

  @inline ∇q = flux_divergence(flux, cache, cell_center_metrics, idx)

  uⁿ = u[idx]
  uⁿ⁻¹ = u_prev[idx]
  du = uⁿ - uⁿ⁻¹
  du = du * !isapprox(uⁿ, uⁿ⁻¹; rtol=ϵ)

  residuals[idx] = -du / dt - ∇q + source_term[idx]
end

"""
    update_residual!(solver::PseudoTransientSolver, mesh, Δt)

"""
function update_residual!(
  solver::PseudoTransientSolver{N,T}, mesh, Δt, ϵ=eps(T)
) where {N,T}
  #
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _update_resid_kernel!(solver.backend)(
    solver.res,
    solver.cache,
    mesh.cell_center_metrics,
    solver.u,
    solver.u_prev,
    solver.q′,
    solver.source_term,
    Δt,
    ϵ,
    idx_offset;
    ndrange=size(domain),
  )

  # KernelAbstractions.synchronize(solver.backend)
  return nothing
end
