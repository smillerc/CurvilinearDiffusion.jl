
include("inner_operators.jl")
include("boundary_operators.jl")
include("metric_terms.jl")

using UnPack
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

  #
  m, _ = size(A)
  mesh_limits = mesh.domain_limits.cell

  kernel = @cuda launch = false assemble_2d_csr_cuda!(
    A,
    scheme.α,
    Δt,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    scheme.halo_aware_indices,
    mesh_limits,
    scheme.mean_func,
  )

  config = launch_configuration(kernel.fun)
  threads = min(m, config.threads)

  blocks = cld(m, threads)

  kernel(
    A,
    scheme.α,
    Δt,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    scheme.halo_aware_indices,
    mesh_limits,
    scheme.mean_func;
    threads,
    blocks,
  )
  # CUDA.synchronize()
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

function assemble_2d_csr_cuda!(
  A, α, Δt, cell_center_metrics, edge_metrics, grid_indices, mesh_limits, meanfunc::F
) where {F}
  # every thread processes an entire row
  row = threadIdx().x + (blockIdx().x - 1) * blockDim().x

  nrows = size(A, 1)
  middle_row = nrows ÷ 2

  row > nrows && return nothing

  @unpack ilo, ihi, jlo, jhi = mesh_limits
  # @inbounds begin
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

  cfirst = A.rowPtr[row]
  clast = A.rowPtr[row + 1] - 1
  ncols = clast - cfirst + 1

  # The matrix will look something like this:
  # ⎡⠻⣦⡙⢷⣄⠀⠀⠀⠀⠀⠀⠀⠀⎤
  # ⎢⢷⣌⠻⣦⡙⢷⣄⠀⠀⠀⠀⠀⠀⎥
  # ⎢⠀⠙⢷⣌⠻⣦⡙⢷⣄⠀⠀⠀⠀⎥
  # ⎢⠀⠀⠀⠙⢷⣌⠻⣦⡙⢷⣄⠀⠀⎥
  # ⎢⠀⠀⠀⠀⠀⠙⢷⣌⠻⣦⡙⢷⡄⎥
  # ⎢⠀⠀⠀⠀⠀⠀⠀⠙⢷⣌⠻⣦⡁⎥
  # ⎣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠁⠈⠁⎦

  upper_half = row < middle_row # are we in the upper half of the matrix?

  # For stencils in the upper half of the matrix (row < middle_row), the nzVals
  # start at 
  starting_col = 9 - ncols

  if i == ilo && j == jlo
    stencil = _neumann_boundary_diffusion_operator(metric_terms, diffusivity, Δt, 5)
  elseif i == ilo && j == jhi
    stencil = _neumann_boundary_diffusion_operator(metric_terms, diffusivity, Δt, 6)
  elseif i == ihi && j == jlo
    stencil = _neumann_boundary_diffusion_operator(metric_terms, diffusivity, Δt, 7)
  elseif i == ihi && j == jhi
    stencil = _neumann_boundary_diffusion_operator(metric_terms, diffusivity, Δt, 8)
  elseif i == ilo
    stencil = _neumann_boundary_diffusion_operator(metric_terms, diffusivity, Δt, 1)
  elseif i == ihi
    stencil = _neumann_boundary_diffusion_operator(metric_terms, diffusivity, Δt, 2)
  elseif j == jlo
    stencil = _neumann_boundary_diffusion_operator(metric_terms, diffusivity, Δt, 3)
  elseif j == jhi
    stencil = _neumann_boundary_diffusion_operator(metric_terms, diffusivity, Δt, 4)
  else
    stencil = inner_op_2d(metric_terms, diffusivity, Δt)
  end

  if ncols == 9 # inner cells
    @inbounds for c in 1:9
      zidx = cfirst + c - 1
      A.nzVal[zidx] = stencil[c]
    end
  else
    if upper_half
      @inbounds for c in 1:ncols
        zidx = cfirst + c - 1
        A.nzVal[zidx] = stencil[starting_col + c]
      end
    else # lower half of the matrix
      @inbounds for c in 1:ncols
        zidx = cfirst + c - 1
        A.nzVal[zidx] = stencil[c]
      end
    end
  end

  # example; we're on row # 2, so ncols = 8, since the first col is chopped off...
  # starting_col = 9 - 8 = 1
  # we're in the upper_half, so we loop 
  # for c in 1:8
  #   A.nzVal[c-1] = stencil[1 + c]
  # end
  # or 
  # A.nzVal[0] = stencil[1 + 1 = 2]
  # A.nzVal[1] = stencil[1 + 2 = 3]
  # A.nzVal[2] = stencil[1 + 3 = 4]
  # A.nzVal[3] = stencil[1 + 4 = 5]
  # A.nzVal[4] = stencil[1 + 5 = 6]
  # A.nzVal[5] = stencil[1 + 6 = 7]
  # A.nzVal[6] = stencil[1 + 7 = 8]
  # A.nzVal[7] = stencil[1 + 8 = 9]

  # starting col is 

  # if ncols == 5
  #   if i == ilo
  #     @inbounds for c in 1:5
  #       A.nzVal[c - 1] = stencil[4 + c]
  #     end
  #   else # i == ihi
  #     @inbounds for c in 1:5
  #       A.nzVal[c - 1] = stencil[c]
  #     end
  #   end
  # elseif ncols == 6
  # elseif ncols == 7
  # elseif ncols == 8
  # elseif ncols == 9 # inner cells
  #   @inbounds for c in 1:9
  #     A.nzVal[c - 1] = stencil[c]
  #   end

  # A.nzVal[cfirst + 0] = stencil[1] or [-1, -1] # (i-1, j-1)
  # A.nzVal[cfirst + 1] = stencil[2] or [+0, -1] # (i  , j-1)
  # A.nzVal[cfirst + 2] = stencil[3] or [+1, -1] # (i+1, j-1)
  # A.nzVal[cfirst + 3] = stencil[4] or [-1, +0] # (i-1, j  )
  # A.nzVal[cfirst + 4] = stencil[5] or [+0, +0] # (i  , j  )
  # A.nzVal[cfirst + 5] = stencil[6] or [+1, +0] # (i+1, j  )
  # A.nzVal[cfirst + 6] = stencil[7] or [-1, +1] # (i-1, j+1)
  # A.nzVal[cfirst + 7] = stencil[8] or [+0, +1] # (i  , j+1)
  # A.nzVal[cfirst + 8] = stencil[9] or [+1, +1] # (i+1, j+1)

  # end

  return nothing
end
