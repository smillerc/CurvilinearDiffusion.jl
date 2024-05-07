
include("stencils.jl")
include("inner_operators.jl")
include("boundary_operators.jl")

function assemble!(
  A::SparseMatrixCSC, u::AbstractArray{T,2}, scheme::ImplicitScheme{2}, mesh, Δt
) where {T}
  backend = scheme.backend
  workgroup = (64,)

  nrows = length(scheme.iterators.full.cartesian)
  _assembly_kernel_2d(backend, workgroup)(
    A,
    scheme.linear_problem.b,
    scheme.source_term,
    u,
    scheme.α,
    Δt,
    mesh.cell_center_metrics.J,
    mesh.edge_metrics,
    scheme.iterators.mesh,
    scheme.iterators.full.cartesian,
    scheme.iterators.full.linear,
    scheme.limits,
    scheme.mean_func,
    scheme.bcs;
    ndrange=nrows,
  )

  KernelAbstractions.synchronize(backend)

  return nothing
end

function assemble!(
  A::SparseMatrixCSC, u::AbstractArray{T,3}, scheme::ImplicitScheme{3}, mesh, Δt
) where {T}
  backend = scheme.backend
  workgroup = (64,)

  nrows = length(scheme.iterators.full.cartesian)
  _assembly_kernel_3d(backend, workgroup)(
    A,
    scheme.linear_problem.b,
    scheme.source_term,
    u,
    scheme.α,
    Δt,
    mesh.cell_center_metrics.J,
    mesh.edge_metrics,
    scheme.iterators.mesh,
    scheme.iterators.full.cartesian,
    scheme.iterators.full.linear,
    scheme.limits,
    scheme.mean_func,
    scheme.bcs;
    ndrange=nrows,
  )

  KernelAbstractions.synchronize(backend)

  return nothing
end
