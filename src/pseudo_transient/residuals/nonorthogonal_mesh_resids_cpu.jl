function update_residuals_nonorthogonal!(
  solver::PseudoTransientSolver{N,T,BE}, mesh, Δt, ϵ=eps(T)
) where {N,T,BE<:CPU}

  #
  domain = solver.iterators.domain.cartesian
  u = solver.u
  u_prev = solver.u_prev
  flux = solver.q′
  cell_center_metrics = mesh.cell_center_metrics
  cache = solver.cache
  residuals = solver.res
  source_term = solver.source_term

  @batch for idx in domain
    @inline ∇q = flux_divergence(flux, cache, cell_center_metrics, idx)

    uⁿ = u[idx]
    uⁿ⁻¹ = u_prev[idx]
    du = uⁿ - uⁿ⁻¹
    du = du * !isapprox(uⁿ, uⁿ⁻¹; rtol=ϵ)

    residuals[idx] = -du / Δt - ∇q + source_term[idx]
  end

  return nothing
end