
# ---------------------------------------------------------------------------
#  Kernels
# ---------------------------------------------------------------------------

# @kernel function inner_diffusion_op_kernel_1d!(
#   A,
#   b,
#   @Const(α),
#   @Const(u),
#   @Const(source_term),
#   @Const(Δt),
#   @Const(cell_center_metrics),
#   @Const(edge_metrics),
#   @Const(grid_indices),
#   @Const(matrix_indices),
#   @Const(mean_func::F),
#   @Const((ni, nj))
# ) where {F}
#   idx = @index(Global, Linear)

#   @inbounds begin
#     grid_idx = grid_indices[idx]

#     i, = grid_idx.I
#     edge_diffusivity = (
#       αᵢ₊½=mean_func(α[i, j], α[i + 1]), αᵢ₋½=mean_func(α[i, j], α[i - 1])
#     )

#     stencil, rhs = _inner_diffusion_operator(
#       u, source_term, edge_diffusivity, Δt, cell_center_metrics, edge_metrics, grid_idx
#     )

#     mat_idx = matrix_indices[idx]
#     #! format: off
#     A[mat_idx, mat_idx - 1]      = stencil[-1] # (i-1, j  )
#     A[mat_idx, mat_idx]          = stencil[+0] # (i  , j  )
#     A[mat_idx, mat_idx + 1]      = stencil[+1] # (i+1, j  )
#     #! format: on

#     b[mat_idx] = rhs
#   end
# end

@kernel function full_diffusion_op_2d!(
  A::SparseMatrixCSC{T,Ti},
  b::AbstractVector{T},
  source_term::AbstractArray{T,N},
  u::AbstractArray{T,N},
  α::AbstractArray{T,N},
  Δt,
  cell_center_jacobian,
  edge_metrics,
  mesh_indices,
  diffusion_prob_indices,
  matrix_indices,
  limits,
  mean_func::F,
  stencil_col_lookup,
  bcs,
) where {T,Ti,N,F<:Function}

  #
  _, ncols = size(A)

  # These are the indicies corresponding to the edge
  # of the diffusion problem
  @unpack ilo, ihi, jlo, jhi = limits

  idx = @index(Global, Linear)

  begin
    row = matrix_indices[idx]
    mesh_idx = mesh_indices[idx]
    diff_idx = diffusion_prob_indices[idx]
    i, j = diff_idx.I

    onbc = i == ilo || i == ihi || j == jlo || j == jhi

    cols = (
      row + first(stencil_col_lookup.ᵢ₋₁ⱼ₋₁), # colᵢ₋₁ⱼ₋₁
      row + first(stencil_col_lookup.ᵢⱼ₋₁),   # colᵢⱼ₋₁
      row + first(stencil_col_lookup.ᵢ₊₁ⱼ₋₁), # colᵢ₊₁ⱼ₋₁
      row + first(stencil_col_lookup.ᵢ₋₁ⱼ),   # colᵢ₋₁ⱼ
      row + first(stencil_col_lookup.ᵢⱼ),     # colᵢⱼ
      row + first(stencil_col_lookup.ᵢ₊₁ⱼ),   # colᵢ₊₁ⱼ
      row + first(stencil_col_lookup.ᵢ₋₁ⱼ₊₁), # colᵢ₋₁ⱼ₊₁
      row + first(stencil_col_lookup.ᵢⱼ₊₁),   # colᵢⱼ₊₁
      row + first(stencil_col_lookup.ᵢ₊₁ⱼ₊₁), # colᵢ₊₁ⱼ₊₁
    )

    if !onbc
      J = cell_center_jacobian[mesh_idx]
      edge_α = edge_diffusivity(α, diff_idx, mean_func)
      A_coeffs = _inner_diffusion_operator(edge_α, Δt, J, edge_metrics, mesh_idx)

      rhs_coeff = (source_term[diff_idx] * Δt + u[mesh_idx]) * J

      for (i, col) in enumerate(cols)
        A[row, col] = A_coeffs[i]
      end
    else
      A_coeffs, rhs_coeff = bc_operator(bcs, diff_idx, limits, T)

      for (i, col) in enumerate(cols)
        if 1 <= col <= ncols
          A[row, col] = A_coeffs[i]
        end
      end
    end

    # @show row, mesh_idx, diff_idx, A_coeffs, rhs_coeff, onbc
    b[row] = rhs_coeff
  end
end

function bc_operator(bcs, idx::CartesianIndex{2}, limits, T)
  @unpack ilo, ihi, jlo, jhi = limits
  i, j = idx.I

  onbc = i == ilo || i == ihi || j == jlo || j == jhi

  if !onbc
    error("The bc_operator is getting called, but we're not on a boundary!")
  end

  at_corner = (
    (i == ihi) && (j == jlo) ||
    (i == ihi) && (j == jhi) ||
    (i == ilo) && (j == jlo) ||
    (i == ilo) && (j == jhi)
  )

  if at_corner
    # return SVector{9,T}(0, 0, 0, 0, 0, 0, 0, 0, 0), zero(T)
    return SVector{9,T}(0, 0, 0, 0, 1, 0, 0, 0, 0), zero(T)
  else
    if i == ilo
      A_coeffs, rhs_coeff = bc_coeffs(bcs.ilo, idx, ILO_BC_LOC, T)
    elseif i == ihi
      A_coeffs, rhs_coeff = bc_coeffs(bcs.ihi, idx, IHI_BC_LOC, T)
    elseif j == jlo
      A_coeffs, rhs_coeff = bc_coeffs(bcs.jlo, idx, JLO_BC_LOC, T)
    elseif j == jhi
      A_coeffs, rhs_coeff = bc_coeffs(bcs.jhi, idx, JHI_BC_LOC, T)
    end
  end

  return A_coeffs, rhs_coeff
end

@kernel function inner_diffusion_op_kernel_2d!(
  A::SparseMatrixCSC{T,Ti},
  b::AbstractVector{T},
  source_term::AbstractArray{T,N},
  u::AbstractArray{T,N},
  α::AbstractArray{T,N},
  Δt,
  cell_center_jacobian,
  edge_metrics,
  grid_indices,
  matrix_indices,
  mean_func::F,
  stencil_col_lookup,
) where {T,Ti,N,F<:Function}
  idx = @index(Global, Linear)

  @inbounds begin
    row = matrix_indices[idx]
    grid_idx = grid_indices[idx]
    J = cell_center_jacobian[grid_idx]
    edge_α = edge_diffusivity(α, grid_idx, mean_func)
    stencil = _inner_diffusion_operator(edge_α, Δt, J, edge_metrics, grid_idx)

    #! format: off
    colᵢ₋₁ⱼ₋₁ = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₋₁)
    colᵢⱼ₋₁ =   row + first(stencil_col_lookup.ᵢⱼ₋₁)
    colᵢ₊₁ⱼ₋₁ = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₋₁)
    colᵢ₋₁ⱼ =   row + first(stencil_col_lookup.ᵢ₋₁ⱼ)
    colᵢⱼ =     row + first(stencil_col_lookup.ᵢⱼ)
    colᵢ₊₁ⱼ =   row + first(stencil_col_lookup.ᵢ₊₁ⱼ)
    colᵢ₋₁ⱼ₊₁ = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₊₁)
    colᵢⱼ₊₁ =   row + first(stencil_col_lookup.ᵢⱼ₊₁)
    colᵢ₊₁ⱼ₊₁ = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₊₁)

    A[row, colᵢ₋₁ⱼ₋₁] = stencil[1]
    A[row, colᵢⱼ₋₁  ] = stencil[2]
    A[row, colᵢ₊₁ⱼ₋₁] = stencil[3]
    A[row, colᵢ₋₁ⱼ  ] = stencil[4]
    A[row, colᵢⱼ    ] = stencil[5]
    A[row, colᵢ₊₁ⱼ  ] = stencil[6]
    A[row, colᵢ₋₁ⱼ₊₁] = stencil[7]
    A[row, colᵢⱼ₊₁  ] = stencil[8]
    A[row, colᵢ₊₁ⱼ₊₁] = stencil[9]
    #! format: on

    # rhs update
    b[row] = (source_term[grid_idx] * Δt + u[grid_idx]) * J
  end
end

@kernel function inner_diffusion_op_kernel_3d!(
  A,
  α,
  Δt,
  cell_center_metrics,
  edge_metrics,
  grid_indices,
  matrix_indices,
  mean_func::F,
  stencil_col_lookup,
) where {F}
  idx = @index(Global, Linear)

  @inbounds begin
    row = matrix_indices[idx]
    grid_idx = grid_indices[idx]
    edge_α = edge_diffusivity(α, grid_idx, mean_func)
    J = cell_center_metrics.J[grid_idx]
    stencil = _inner_diffusion_operator(edge_α, Δt, J, edge_metrics, grid_idx)

    cols = (
      row + first(stencil_col_lookup.ᵢ₋₁ⱼ₋₁ₖ₋₁), # colᵢ₋₁ⱼ₋₁ₖ₋₁
      row + first(stencil_col_lookup.ᵢⱼ₋₁ₖ₋₁),   # colᵢⱼ₋₁ₖ₋₁
      row + first(stencil_col_lookup.ᵢ₊₁ⱼ₋₁ₖ₋₁), # colᵢ₊₁ⱼ₋₁ₖ₋₁
      row + first(stencil_col_lookup.ᵢ₋₁ⱼₖ₋₁),   # colᵢ₋₁ⱼₖ₋₁
      row + first(stencil_col_lookup.ᵢⱼₖ₋₁),     # colᵢⱼₖ₋₁
      row + first(stencil_col_lookup.ᵢ₊₁ⱼₖ₋₁),   # colᵢ₊₁ⱼₖ₋₁
      row + first(stencil_col_lookup.ᵢ₋₁ⱼ₊₁ₖ₋₁), # colᵢ₋₁ⱼ₊₁ₖ₋₁
      row + first(stencil_col_lookup.ᵢⱼ₊₁ₖ₋₁),   # colᵢⱼ₊₁ₖ₋₁
      row + first(stencil_col_lookup.ᵢ₊₁ⱼ₊₁ₖ₋₁), # colᵢ₊₁ⱼ₊₁ₖ₋₁
      row + first(stencil_col_lookup.ᵢ₋₁ⱼ₋₁ₖ),   # colᵢ₋₁ⱼ₋₁ₖ
      row + first(stencil_col_lookup.ᵢⱼ₋₁ₖ),     # colᵢⱼ₋₁ₖ
      row + first(stencil_col_lookup.ᵢ₊₁ⱼ₋₁ₖ),   # colᵢ₊₁ⱼ₋₁ₖ
      row + first(stencil_col_lookup.ᵢ₋₁ⱼₖ),     # colᵢ₋₁ⱼₖ
      row + first(stencil_col_lookup.ᵢⱼₖ),       # colᵢⱼₖ
      row + first(stencil_col_lookup.ᵢ₊₁ⱼₖ),     # colᵢ₊₁ⱼₖ
      row + first(stencil_col_lookup.ᵢ₋₁ⱼ₊₁ₖ),   # colᵢ₋₁ⱼ₊₁ₖ
      row + first(stencil_col_lookup.ᵢⱼ₊₁ₖ),     # colᵢⱼ₊₁ₖ
      row + first(stencil_col_lookup.ᵢ₊₁ⱼ₊₁ₖ),   # colᵢ₊₁ⱼ₊₁ₖ
      row + first(stencil_col_lookup.ᵢ₋₁ⱼ₋₁ₖ₊₁), # colᵢ₋₁ⱼ₋₁ₖ₊₁
      row + first(stencil_col_lookup.ᵢⱼ₋₁ₖ₊₁),   # colᵢⱼ₋₁ₖ₊₁
      row + first(stencil_col_lookup.ᵢ₊₁ⱼ₋₁ₖ₊₁), # colᵢ₊₁ⱼ₋₁ₖ₊₁
      row + first(stencil_col_lookup.ᵢ₋₁ⱼₖ₊₁),   # colᵢ₋₁ⱼₖ₊₁
      row + first(stencil_col_lookup.ᵢⱼₖ₊₁),     # colᵢⱼₖ₊₁
      row + first(stencil_col_lookup.ᵢ₊₁ⱼₖ₊₁),   # colᵢ₊₁ⱼₖ₊₁
      row + first(stencil_col_lookup.ᵢ₋₁ⱼ₊₁ₖ₊₁), # colᵢ₋₁ⱼ₊₁ₖ₊₁
      row + first(stencil_col_lookup.ᵢⱼ₊₁ₖ₊₁),   # colᵢⱼ₊₁ₖ₊₁
      row + first(stencil_col_lookup.ᵢ₊₁ⱼ₊₁ₖ₊₁), # colᵢ₊₁ⱼ₊₁ₖ₊₁
    )

    @inbounds for (c, col) in enumerate(cols)
      A[row, col] = stencil[c]
    end
  end
end

# ---------------------------------------------------------------------------
#  Edge diffusivity
# ---------------------------------------------------------------------------

function edge_diffusivity(α, idx::CartesianIndex{1}, mean_function::F) where {F<:Function}
  i, = idx.I
  edge_diffusivity = (
    αᵢ₊½=mean_function(α[i], α[i + 1]), #
    αᵢ₋½=mean_function(α[i], α[i - 1]), #
  )

  return edge_diffusivity
end

function edge_diffusivity(α, idx::CartesianIndex{2}, mean_function::F) where {F<:Function}
  i, j = idx.I
  edge_diffusivity = (
    αᵢ₊½=mean_function(α[i, j], α[i + 1, j]),
    αᵢ₋½=mean_function(α[i, j], α[i - 1, j]),
    αⱼ₊½=mean_function(α[i, j], α[i, j + 1]),
    αⱼ₋½=mean_function(α[i, j], α[i, j - 1]),
  )

  return edge_diffusivity
end

function edge_diffusivity(α, idx::CartesianIndex{3}, mean_function::F) where {F<:Function}
  i, j, k = idx.I
  edge_diffusivity = (
    αᵢ₊½=mean_function(α[i, j, k], α[i + 1, j, k]),
    αᵢ₋½=mean_function(α[i, j, k], α[i - 1, j, k]),
    αⱼ₊½=mean_function(α[i, j, k], α[i, j + 1, k]),
    αⱼ₋½=mean_function(α[i, j, k], α[i, j - 1, k]),
    αₖ₊½=mean_function(α[i, j, k], α[i, j, k + 1]),
    αₖ₋½=mean_function(α[i, j, k], α[i, j, k - 1]),
  )

  return edge_diffusivity
end

# ---------------------------------------------------------------------------
#  Inner-domain Operators
# ---------------------------------------------------------------------------

# Generate a stencil for a single 1D cell in the interior
@inline function _inner_diffusion_operator(
  edge_diffusivity, Δt, J, edge_metrics, idx::CartesianIndex{1}
)
  edge_terms = conservative_edge_terms(edge_diffusivity, edge_metrics, idx)
  stencil = stencil_1d(edge_terms, J, Δt)
  return stencil
end

# Generate a stencil for a single 2D cell in the interior
@inline function _inner_diffusion_operator(
  edge_diffusivity, Δt, J, edge_metrics, idx::CartesianIndex{2}
)
  edge_terms = conservative_edge_terms(edge_diffusivity, edge_metrics, idx)
  stencil = stencil_2d(edge_terms, J, Δt)
  return stencil
end

# Generate a stencil for a single 3D cell in the interior
@inline function _inner_diffusion_operator(
  edge_diffusivity, Δt, J, edge_metrics, idx::CartesianIndex{3}
)
  edge_terms = conservative_edge_terms(edge_diffusivity, edge_metrics, idx)
  stencil = stencil_3d(edge_terms, J, Δt)
  return stencil
end

# ---------------------------------------------------------------------------
#  Boundary Operators
# ---------------------------------------------------------------------------

# # Generate a stencil for a single 2D cell in the interior
# @inline function _bc_diffusion_operator(
#   ::NeumannBC, edge_diffusivity, Δt, J, edge_metrics, idx::CartesianIndex{2}, mesh_limits
# )
#   edge_terms = conservative_edge_terms(edge_diffusivity, edge_metrics, idx)
#   stencil = neumann_stencil_2d(edge_terms, J, Δt, idx, mesh_limits)
#   return stencil
# end
