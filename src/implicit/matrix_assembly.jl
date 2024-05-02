
include("stencils.jl")
include("inner_operators.jl")
include("boundary_operators.jl")

"""
    assemble_matrix!(scheme::ImplicitScheme, u, Δt)

Assemble the `A` matrix and right-hand side vector `b` for the solution
to the 2D diffusion problem for a state-array `u` over a time step `Δt`.
"""

function assemble_matrix!(A::SparseMatrixCSC, scheme::ImplicitScheme{2}, mesh, Δt)
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
    A,
    scheme.α,
    Δt,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    inner_domain,
    matrix_indices,
    scheme.mean_func,
    scheme.stencil_col_lookup;
    ndrange=size(inner_domain),
  )

  bc_ops = (
    bc_operator(scheme.bcs.ilo),
    bc_operator(scheme.bcs.ihi),
    bc_operator(scheme.bcs.jlo),
    bc_operator(scheme.bcs.jhi),
  )

  bc_domains = (
    @view(scheme.halo_aware_indices[begin, :]), # ilo
    @view(scheme.halo_aware_indices[end, :]),   # ihi
    @view(scheme.halo_aware_indices[:, begin]), # jlo
    @view(scheme.halo_aware_indices[:, end]),   # jhi
  )

  DI = LinearIndices(scheme.domain_indices)
  bc_matrix_indices = (
    @view(DI[begin, :]), @view(DI[end, :]), @view(DI[:, begin]), @view(DI[:, end])
  )

  for bc in 1:4
    boundary_diffusion_op_kernel_2d!(backend, workgroup)(
      A,
      scheme.α,
      Δt,
      mesh.cell_center_metrics,
      mesh.edge_metrics,
      bc_domains[bc],
      bc_matrix_indices[bc],
      scheme.mean_func,
      scheme.stencil_col_lookup,
      bc_ops[bc],
      bc;
      ndrange=size(bc_domains[bc]),
    )
  end

  KernelAbstractions.synchronize(backend)

  return nothing
end

function assemble_matrix!(A::SparseMatrixCSC, scheme::ImplicitScheme{3}, mesh, Δt)
  nhalo = 1
  matrix_domain_LI = LinearIndices(scheme.domain_indices)

  matrix_indices = @view matrix_domain_LI[
    (nhalo + 1):(end - nhalo), (nhalo + 1):(end - nhalo), (nhalo + 1):(end - nhalo)
  ]

  inner_domain = @view scheme.halo_aware_indices[
    (nhalo + 1):(end - nhalo), (nhalo + 1):(end - nhalo), (nhalo + 1):(end - nhalo)
  ]

  backend = scheme.backend
  workgroup = (64,)

  inner_diffusion_op_kernel_3d!(backend, workgroup)(
    A,
    scheme.α,
    Δt,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    inner_domain,
    matrix_indices,
    scheme.mean_func,
    scheme.stencil_col_lookup;
    ndrange=size(inner_domain),
  )

  bc_ops = (
    bc_operator(scheme.bcs.ilo),
    bc_operator(scheme.bcs.ihi),
    bc_operator(scheme.bcs.jlo),
    bc_operator(scheme.bcs.jhi),
    bc_operator(scheme.bcs.klo),
    bc_operator(scheme.bcs.khi),
  )

  bc_domains = (
    @view(scheme.halo_aware_indices[begin, :, :]), # ilo
    @view(scheme.halo_aware_indices[end, :, :]),   # ihi
    @view(scheme.halo_aware_indices[:, begin, :]), # jlo
    @view(scheme.halo_aware_indices[:, end, :]),   # jhi
    @view(scheme.halo_aware_indices[:, :, begin]), # klo
    @view(scheme.halo_aware_indices[:, :, end]),   # khi
  )

  DI = LinearIndices(scheme.domain_indices)
  bc_matrix_indices = (
    @view(DI[begin, :, :]),
    @view(DI[end, :, :]),
    @view(DI[:, begin, :]),
    @view(DI[:, end, :]),
    @view(DI[:, :, begin]),
    @view(DI[:, :, end])
  )

  for bc in 1:6
    boundary_diffusion_op_kernel_3d!(backend, workgroup)(
      A,
      scheme.α,
      Δt,
      mesh.cell_center_metrics,
      mesh.edge_metrics,
      bc_domains[bc],
      bc_matrix_indices[bc],
      scheme.mean_func,
      scheme.stencil_col_lookup,
      bc_ops[bc],
      bc;
      ndrange=size(bc_domains[bc]),
    )
  end

  KernelAbstractions.synchronize(backend)

  return nothing
end
