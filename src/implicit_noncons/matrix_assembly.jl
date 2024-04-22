
include("inner_operators.jl")
include("boundary_operators.jl")
include("metric_terms.jl")

using CUDA.CUSPARSE: CSRIterator, _getindex

"""
    assemble_matrix!(scheme::ImplicitScheme,  Δt)

Assemble the `A` matrix and right-hand side vector `b` for the solution
to the 2D diffusion problem for a state-array `u` over a time step `Δt`.
"""

# function assemble_matrix!(scheme::ImplicitScheme{2}, A, mesh, Δt)
#   ni, nj = size(scheme.domain_indices)

#   nhalo = 1
#   matrix_domain_LI = LinearIndices(scheme.domain_indices)

#   matrix_indices = @view matrix_domain_LI[
#     (nhalo + 1):(end - nhalo), (nhalo + 1):(end - nhalo)
#   ]

#   inner_domain = @view scheme.halo_aware_indices[
#     (nhalo + 1):(end - nhalo), (nhalo + 1):(end - nhalo)
#   ]

#   backend = CPU() #scheme.backend
#   workgroup = (64,)

#   inner_diffusion_op_kernel_2d!(backend, workgroup)(
#     A,
#     scheme.α,
#     Δt,
#     mesh.cell_center_metrics,
#     mesh.edge_metrics,
#     inner_domain,
#     matrix_indices,
#     scheme.mean_func,
#     (ni, nj);
#     ndrange=size(inner_domain),
#   )

#   bc_locs = (ilo=1, ihi=2, jlo=3, jhi=4)
#   # ilo
#   ilo_domain = @view scheme.halo_aware_indices[begin, :]
#   ilo_matrix_indices = @view LinearIndices(scheme.domain_indices)[begin, :]
#   boundary_diffusion_op_kernel_2d!(backend, workgroup)(
#     A,
#     scheme.α,
#     Δt,
#     mesh.cell_center_metrics,
#     mesh.edge_metrics,
#     ilo_domain,
#     ilo_matrix_indices,
#     scheme.mean_func,
#     (ni, nj),
#     bc_locs.ilo;
#     ndrange=size(ilo_domain),
#   )

#   # ihi
#   ihi_domain = @view scheme.halo_aware_indices[end, :]
#   ihi_matrix_indices = @view LinearIndices(scheme.domain_indices)[end, :]
#   boundary_diffusion_op_kernel_2d!(backend, workgroup)(
#     A,
#     scheme.α,
#     Δt,
#     mesh.cell_center_metrics,
#     mesh.edge_metrics,
#     ihi_domain,
#     ihi_matrix_indices,
#     scheme.mean_func,
#     (ni, nj),
#     bc_locs.ihi;
#     ndrange=size(ihi_domain),
#   )

#   # jlo
#   jlo_domain = @view scheme.halo_aware_indices[:, begin]
#   jlo_matrix_indices = @view LinearIndices(scheme.domain_indices)[:, begin]
#   boundary_diffusion_op_kernel_2d!(backend, workgroup)(
#     A,
#     scheme.α,
#     Δt,
#     mesh.cell_center_metrics,
#     mesh.edge_metrics,
#     jlo_domain,
#     jlo_matrix_indices,
#     scheme.mean_func,
#     (ni, nj),
#     bc_locs.jlo;
#     ndrange=size(jlo_domain),
#   )

#   # jhi
#   jhi_domain = @view scheme.halo_aware_indices[:, end]
#   jhi_matrix_indices = @view LinearIndices(scheme.domain_indices)[:, end]
#   boundary_diffusion_op_kernel_2d!(backend, workgroup)(
#     A,
#     scheme.α,
#     Δt,
#     mesh.cell_center_metrics,
#     mesh.edge_metrics,
#     jhi_domain,
#     jhi_matrix_indices,
#     scheme.mean_func,
#     (ni, nj),
#     bc_locs.jhi;
#     ndrange=size(jhi_domain),
#   )

#   KernelAbstractions.synchronize(backend)

#   return nothing
# end

function assemble_matrix!(
  scheme::ImplicitScheme{2,T,BE}, A::CuSparseMatrixCSR{Tv,Ti}, mesh, Δt
) where {Tv,Ti,T,BE}
  # nrows, ncols = A.dims
  # _assemble(scheme, A.rowPtr, A.nzVal, nrows, mesh, Δt)
  _assemble(scheme, mesh, Δt)
end

# function _assemble(
#   scheme::ImplicitScheme{2,T,BE}, rowptr, nzval, nrows, mesh, Δt
# ) where {T,BE}
#   backend = scheme.backend

#   nhalo = 1
#   inner_domain = @view scheme.halo_aware_indices[
#     (nhalo + 1):(end - nhalo), (nhalo + 1):(end - nhalo)
#   ]

#   workgroup = (64,)
#   assemble_csr_kernel!(backend)(
#     rowptr,
#     nzval,
#     scheme.α,
#     Δt,
#     mesh.cell_center_metrics,
#     mesh.edge_metrics,
#     scheme.halo_aware_indices,
#     scheme.mean_func;
#     ndrange=nrows,
#   )
#   return KernelAbstractions.synchronize(backend)
# end

function _assemble(scheme::ImplicitScheme{2,T,BE}, mesh, Δt) where {T,BE}

  #
  m, _ = size(scheme.linear_problem.A)

  kernel = @cuda launch = false assemble_csr_cuda!(
    scheme.linear_problem.A,
    scheme.α,
    Δt,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    scheme.halo_aware_indices,
    scheme.mean_func,
  )

  config = launch_configuration(kernel.fun)
  threads = min(m, config.threads)
  blocks = cld(m, threads)

  kernel(
    scheme.linear_problem.A,
    scheme.α,
    Δt,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    scheme.halo_aware_indices,
    scheme.mean_func;
    threads,
    blocks,
  )

  CUDA.synchronize()

  return nothing
end

@kernel function assemble_csr_kernel!(
  rowPtr, nzVal, α, Δt, cell_center_metrics, edge_metrics, grid_indices, meanfunc::F
) where {F}
  #
  row = @index(Global, Linear)

  # @inbounds begin
  # grid_idx = grid_indices[row]
  # i, j = grid_idx.I
  # metric_terms = non_cons_terms(cell_center_metrics, edge_metrics, grid_idx)

  # aᵢⱼ = α[i, j]
  # aᵢ₊₁ⱼ = α[i + 1, j]
  # aᵢ₋₁ⱼ = α[i - 1, j]
  # aᵢⱼ₊₁ = α[i, j + 1]
  # aᵢⱼ₋₁ = α[i, j - 1]

  # diffusivity = (;
  #   aᵢⱼ,
  #   aᵢ₊₁ⱼ,
  #   aᵢ₋₁ⱼ,
  #   aᵢⱼ₊₁,
  #   aᵢⱼ₋₁,
  #   aᵢ₊½=meanfunc(aᵢⱼ, aᵢ₊₁ⱼ),
  #   aᵢ₋½=meanfunc(aᵢⱼ, aᵢ₋₁ⱼ),
  #   aⱼ₊½=meanfunc(aᵢⱼ, aᵢⱼ₊₁),
  #   aⱼ₋½=meanfunc(aᵢⱼ, aᵢⱼ₋₁),
  # )

  # stencil = inner_op_2d(metric_terms, diffusivity, Δt)

  # # loop through the colums
  # col_iter = rowPtr[row]:(rowPtr[row + 1] - 1)
  # c = rowPtr[row]
  # ncols = length(col_iter)

  # if ncols == 9
  #   nzVal[c + 0] = stencil[-1, -1] # (i-1, j-1)
  #   nzVal[c + 1] = stencil[+0, -1] # (i  , j-1)
  #   nzVal[c + 2] = stencil[+1, -1] # (i+1, j-1)
  #   nzVal[c + 3] = stencil[-1, +0] # (i-1, j  )
  #   nzVal[c + 4] = stencil[+0, +0] # (i  , j  )
  #   nzVal[c + 5] = stencil[+1, +0] # (i+1, j  )
  #   nzVal[c + 6] = stencil[-1, +1] # (i-1, j+1)
  #   nzVal[c + 7] = stencil[+0, +1] # (i  , j+1)
  #   nzVal[c + 8] = stencil[+1, +1] # (i+1, j+1)
  #   # else
  #   # @print("ncols $ncols $stencil \n")
  # end
  # end
end

function assemble_csr_cuda!(
  A, α, Δt, cell_center_metrics, edge_metrics, grid_indices, meanfunc::F
) where {F}
  # every thread processes an entire row
  row = threadIdx().x + (blockIdx().x - 1) * blockDim().x
  iter = @inbounds CSRIterator{Int}(row, A)

  @inbounds begin
    grid_idx = grid_indices[row]
    i, j = grid_idx.I
    metric_terms = non_cons_terms(cell_center_metrics, edge_metrics, grid_idx)

    aᵢⱼ = α[i, j]
    aᵢ₊₁ⱼ = α[i + 1, j]
    aᵢ₋₁ⱼ = α[i - 1, j]
    aᵢⱼ₊₁ = α[i, j + 1]
    aᵢⱼ₋₁ = α[i, j - 1]

    diffusivity = (;
      aᵢⱼ,
      aᵢ₊₁ⱼ,
      aᵢ₋₁ⱼ,
      aᵢⱼ₊₁,
      aᵢⱼ₋₁,
      aᵢ₊½=meanfunc(aᵢⱼ, aᵢ₊₁ⱼ),
      aᵢ₋½=meanfunc(aᵢⱼ, aᵢ₋₁ⱼ),
      aⱼ₊½=meanfunc(aᵢⱼ, aᵢⱼ₊₁),
      aⱼ₋½=meanfunc(aᵢⱼ, aᵢⱼ₋₁),
    )

    stencil = inner_op_2d(metric_terms, diffusivity, Δt)
  end

  # loop through the colums
  # col_iter = A.rowPtr[row]:(A.rowPtr[row + 1] - 1)
  # c = rowPtr[row]
  # ncols = length(col_iter)

  # # if ncols == 9
  # #   nzVal[c + 0] = stencil[-1, -1] # (i-1, j-1)
  # #   nzVal[c + 1] = stencil[+0, -1] # (i  , j-1)
  # #   nzVal[c + 2] = stencil[+1, -1] # (i+1, j-1)
  # #   nzVal[c + 3] = stencil[-1, +0] # (i-1, j  )
  # #   nzVal[c + 4] = stencil[+0, +0] # (i  , j  )
  # #   nzVal[c + 5] = stencil[+1, +0] # (i+1, j  )
  # #   nzVal[c + 6] = stencil[-1, +1] # (i-1, j+1)
  # #   nzVal[c + 7] = stencil[+0, +1] # (i  , j+1)
  # #   nzVal[c + 8] = stencil[+1, +1] # (i+1, j+1)
  # # end

  # ncols = zero(Int32, 0)
  # for (col, ptr) in iter
  #   ncols += 1
  # end
  # # reduce the values for this row
  for (col, ptr) in iter
    I = CartesianIndex(row, col)
    # c = _getindex(A, I, ptr)
    # vals = ntuple(Val(length(args))) do i
    #   arg = @inbounds args[i]
    # end
  end

  return nothing
end

# function nzval