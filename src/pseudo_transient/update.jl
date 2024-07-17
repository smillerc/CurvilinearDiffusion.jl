#
@kernel function _update_kernel!(
  H::AbstractArray{T,2},
  H_prev,
  cell_center_metrics, # applying @Const to a struct array causes problems
  edge_metrics, # applying @Const to a struct array causes problems
  α,
  q,
  dτ_ρ,
  source_term,
  dt,
  I0,
) where {T}
  idx = @index(Global, Cartesian)
  idx += I0

  qᵢ = q.x
  qⱼ = q.y

  @inbounds begin
    ∇q = flux_divergence((qᵢ, qⱼ), H, α, cell_center_metrics, edge_metrics, idx)

    H[idx] = (
      (H[idx] + dτ_ρ[idx] * (H_prev[idx] / dt - ∇q + source_term[idx])) /
      (1 + dτ_ρ[idx] / dt)
    )
  end
end

"""
"""
function compute_update!(solver, mesh, Δt)
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _update_kernel!(solver.backend)(
    solver.H,
    solver.H_prev,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    solver.α,
    solver.qH,
    solver.dτ_ρ,
    solver.source_term,
    Δt,
    idx_offset;
    ndrange=size(domain),
  )

  KernelAbstractions.synchronize(solver.backend)
  return nothing
end