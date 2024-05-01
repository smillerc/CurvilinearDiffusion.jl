
"""
Assemble the right-hand-side, or `b`, in the system `Ax⃗=b`.
"""
function assemble_rhs!(rhs, scheme, mesh, u, Δt)
  matrix_indices = LinearIndices(scheme.domain_indices)
  grid_indices = scheme.halo_aware_indices

  backend = scheme.backend

  rhs_kernel!(backend)(
    rhs,
    u,
    scheme.source_term,
    mesh.cell_center_metrics.J, # cell-centered jacobian
    Δt,
    grid_indices,
    matrix_indices;
    ndrange=size(grid_indices),
  )

  return nothing
end

@kernel function rhs_kernel!(rhs, u, source_term, J, Δt, grid_indices, matrix_indices)
  idx = @index(Global, Linear)

  @inbounds begin
    grid_idx = grid_indices[idx]
    mat_idx = matrix_indices[idx]
    # rhs[mat_idx] = -(J[grid_idx] * source_term[grid_idx] + u[grid_idx] * J[grid_idx] / Δt)
    rhs[mat_idx] = -(J[grid_idx] * source_term[grid_idx] * Δt + u[grid_idx] * J[grid_idx])
  end
end