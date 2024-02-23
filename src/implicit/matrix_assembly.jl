
include("inner_operators.jl")
include("boundary_operators.jl")

"""
    assemble_matrix!(scheme::ImplicitScheme, u, Δt)

Assemble the `A` matrix and right-hand side vector `b` for the solution
to the 2D diffusion problem for a state-array `u` over a time step `Δt`.
"""

function assemble_matrix!(scheme::ImplicitScheme{2}, mesh, u, Δt)
  ni, nj = size(scheme.domain_indices)

  nhalo = 1
  matrix_domain_LI = LinearIndices(scheme.domain_indices)

  matrix_indices = @view matrix_domain_LI[
    (nhalo + 1):(end - nhalo), (nhalo + 1):(end - nhalo)
  ]

  inner_domain = @view scheme.halo_aware_indices[
    (nhalo + 1):(end - nhalo), (nhalo + 1):(end - nhalo)
  ]

  backend = scheme.backend
  workgroup = (64,)

  inner_diffusion_op_kernel_2d!(backend, workgroup)(
    scheme.A,
    scheme.b,
    scheme.α,
    u,
    scheme.source_term,
    Δt,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    inner_domain,
    matrix_indices,
    scheme.mean_func,
    (ni, nj);
    ndrange=size(inner_domain),
  )

  # ilo
  ilo_domain = @view scheme.halo_aware_indices[begin, :]
  ilo_matrix_indices = @view LinearIndices(scheme.domain_indices)[begin, :]
  boundary_diffusion_op_kernel_2d!(backend, workgroup)(
    scheme.A,
    scheme.b,
    scheme.α,
    u,
    scheme.source_term,
    Δt,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    ilo_domain,
    ilo_matrix_indices,
    scheme.mean_func,
    (ni, nj),
    :ilo;
    ndrange=size(ilo_domain),
  )

  # ihi
  ihi_domain = @view scheme.halo_aware_indices[end, :]
  ihi_matrix_indices = @view LinearIndices(scheme.domain_indices)[end, :]
  boundary_diffusion_op_kernel_2d!(backend, workgroup)(
    scheme.A,
    scheme.b,
    scheme.α,
    u,
    scheme.source_term,
    Δt,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    ihi_domain,
    ihi_matrix_indices,
    scheme.mean_func,
    (ni, nj),
    :ihi;
    ndrange=size(ihi_domain),
  )

  # jlo
  jlo_domain = @view scheme.halo_aware_indices[:, begin]
  jlo_matrix_indices = @view LinearIndices(scheme.domain_indices)[:, begin]
  boundary_diffusion_op_kernel_2d!(backend, workgroup)(
    scheme.A,
    scheme.b,
    scheme.α,
    u,
    scheme.source_term,
    Δt,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    jlo_domain,
    jlo_matrix_indices,
    scheme.mean_func,
    (ni, nj),
    :jlo;
    ndrange=size(jlo_domain),
  )

  # jhi
  jhi_domain = @view scheme.halo_aware_indices[:, end]
  jhi_matrix_indices = @view LinearIndices(scheme.domain_indices)[:, end]
  boundary_diffusion_op_kernel_2d!(backend, workgroup)(
    scheme.A,
    scheme.b,
    scheme.α,
    u,
    scheme.source_term,
    Δt,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    jhi_domain,
    jhi_matrix_indices,
    scheme.mean_func,
    (ni, nj),
    :jhi;
    ndrange=size(jhi_domain),
  )

  KernelAbstractions.synchronize(backend)

  return nothing
end
