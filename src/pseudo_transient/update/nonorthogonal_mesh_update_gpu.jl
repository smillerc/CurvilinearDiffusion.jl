function update_nonorthogonal!(
  solver::PseudoTransientSolver{1,T,BE}, mesh, Δt
) where {T,BE<:GPU}
  update_orthogonal!(solver, mesh, Δt) # 1D is always orthogonal
end

function update_nonorthogonal!(
  solver::PseudoTransientSolver{N,T,BE}, mesh, Δt
) where {N,T,BE<:GPU}
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _update_nonorth_kernel!(solver.backend)(
    solver.u,
    solver.u_prev,
    solver.cache,
    mesh.cell_center_metrics,
    solver.q,
    solver.dτ_ρ,
    solver.source_term,
    Δt,
    idx_offset;
    ndrange=size(domain),
  )

  KernelAbstractions.synchronize(solver.backend)
  return nothing
end

@kernel inbounds = true function _update_nonorth_kernel!(
  u,
  @Const(u_prev),
  cache,
  cell_center_metrics, # applying @Const to a struct array causes problems
  @Const(flux),
  @Const(dτ_ρ),
  @Const(source_term),
  @Const(dt),
  @Const(I0),
)
  idx = @index(Global, Cartesian)
  idx += I0

  @inline ∇q = flux_divergence(flux, cache, cell_center_metrics, idx)

  u[idx] = (
    (u[idx] + dτ_ρ[idx] * (u_prev[idx] / dt - ∇q + source_term[idx])) / (1 + dτ_ρ[idx] / dt)
  )
end
