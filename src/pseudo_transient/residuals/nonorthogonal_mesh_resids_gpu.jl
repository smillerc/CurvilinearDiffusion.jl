@kernel inbounds = true function _update_resid!(
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

function update_residuals_nonorthogonal!(
  solver::PseudoTransientSolver{N,T,BE}, mesh, Δt, ϵ=eps(T)
) where {N,T,BE<:GPU}

  #
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _update_resid!(solver.backend)(
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