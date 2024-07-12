@kernel function _update_kernel_arb_mesh!(
  u,
  u_prev,
  cell_center_metrics, # applying @Const to a struct array causes problems
  edge_metrics, # applying @Const to a struct array causes problems
  α,
  q,
  dτ_ρ,
  source_term,
  dt,
  index_offset,
)
  idx = @index(Global, Cartesian)
  idx += index_offset

  @inbounds begin
    @inline ∇q = _arbitrary_flux_divergence(q, u, α, cell_center_metrics, edge_metrics, idx)

    u[idx] = (
      (u[idx] + dτ_ρ[idx] * (u_prev[idx] / dt - ∇q + source_term[idx])) /
      (1 + dτ_ρ[idx] / dt)
    )
  end
end

@kernel function _update_kernel_orth_mesh!(
  u,
  u_prev,
  cell_center_metrics, # applying @Const to a struct array causes problems
  α,
  q,
  dτ_ρ,
  source_term,
  dt,
  index_offset,
)
  idx = @index(Global, Cartesian)
  idx += index_offset

  @inbounds begin
    @inline ∇q = _orthogonal_flux_divergence(q, u, α, cell_center_metrics, idx)

    u[idx] = (
      (u[idx] + dτ_ρ[idx] * (u_prev[idx] / dt - ∇q + source_term[idx])) /
      (1 + dτ_ρ[idx] / dt)
    )
  end
end

function compute_update!(solver, mesh, Δt)
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  if mesh.is_orthogonal
    _update_kernel_orth_mesh!(solver.backend)(
      solver.u,
      solver.u_prev,
      mesh.cell_center_metrics,
      solver.α,
      solver.q,
      solver.dτ_ρ,
      solver.source_term,
      Δt,
      idx_offset;
      ndrange=size(domain),
    )
  else
    _update_kernel_arb_mesh!(solver.backend)(
      solver.u,
      solver.u_prev,
      mesh.cell_center_metrics,
      mesh.edge_metrics,
      solver.α,
      solver.q,
      solver.dτ_ρ,
      solver.source_term,
      Δt,
      idx_offset;
      ndrange=size(domain),
    )
  end

  KernelAbstractions.synchronize(solver.backend)
  return nothing
end
