
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
  grid_indices,
  matrix_indices,
  mesh_limits,
  mean_func::F,
  stencil_col_lookup,
) where {T,Ti,N,F<:Function}

  #
  @unpack ilo, ihi, jlo, jhi = mesh_limits

  idx = @index(Global, Linear)

  @inbounds begin
    row = matrix_indices[idx]
    grid_idx = grid_indices[idx]
    J = cell_center_jacobian[grid_idx]
    edge_α = edge_diffusivity(α, grid_idx, mean_func)
    stencil = diffusion_operator(edge_α, Δt, J, edge_metrics, grid_idx)

    colᵢ₋₁ⱼ₋₁ = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₋₁)
    colᵢⱼ₋₁ = row + first(stencil_col_lookup.ᵢⱼ₋₁)
    colᵢ₊₁ⱼ₋₁ = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₋₁)
    colᵢ₋₁ⱼ = row + first(stencil_col_lookup.ᵢ₋₁ⱼ)
    colᵢⱼ = row + first(stencil_col_lookup.ᵢⱼ)
    colᵢ₊₁ⱼ = row + first(stencil_col_lookup.ᵢ₊₁ⱼ)
    colᵢ₋₁ⱼ₊₁ = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₊₁)
    colᵢⱼ₊₁ = row + first(stencil_col_lookup.ᵢⱼ₊₁)
    colᵢ₊₁ⱼ₊₁ = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₊₁)

    onbc = i == ilo || i == ihi || j == jlo || j == jhi

    if onbc
      A_coeffs, rhs_coeffs = bc_operator(edge_terms, J, Δt, idx, mesh_limits)
    else
      A_coeffs, rhs_coeffs = inner_operator(edge_α, Δt, J, edge_metrics, grid_idx)
    end

    A[row, colᵢ₋₁ⱼ₋₁] = stencil[1]
    A[row, colᵢⱼ₋₁] = stencil[2]
    A[row, colᵢ₊₁ⱼ₋₁] = stencil[3]
    A[row, colᵢ₋₁ⱼ] = stencil[4]
    A[row, colᵢⱼ] = stencil[5]
    A[row, colᵢ₊₁ⱼ] = stencil[6]
    A[row, colᵢ₋₁ⱼ₊₁] = stencil[7]
    A[row, colᵢⱼ₊₁] = stencil[8]
    A[row, colᵢ₊₁ⱼ₊₁] = stencil[9]

    # rhs update
    b[row] = (source_term[grid_idx] * Δt + u[grid_idx]) * J
  end
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
#  RHS update
# ---------------------------------------------------------------------------

function rhs_update(source_term, Δt, u, J)
  return (source_term * Δt + u) * J
end

function rhs_update(
  bc::DirichletBC, stencil, source_term, Δt, u, J, idx::CartesianIndex{2}, mesh_limits
)
  @unpack ilo, ihi, jlo, jhi = mesh_limits

  i, j = idx.I

  uᵢ₋₁ⱼ₋₁, uᵢⱼ₋₁, uᵢ₊₁ⱼ₋₁, uᵢ₋₁ⱼ, uᵢⱼ, uᵢ₊₁ⱼ, uᵢ₋₁ⱼ₊₁, uᵢⱼ₊₁, uᵢ₊₁ⱼ₊₁ = stencil

  u_bc = bc.val # boundary condition value, e.g. boundary temperature

  rhs = (source_term * Δt + u) * J

  if i == ilo
    rhs += -(u_bc * uᵢ₋₁ⱼ)
  end

  if j == jlo
  end

  if i == ihi
  end

  if j == jhi
  end

  return rhs
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

# Generate a stencil for a single 2D cell in the interior
@inline function _bc_diffusion_operator(
  ::NeumannBC, edge_diffusivity, Δt, J, edge_metrics, idx::CartesianIndex{2}, mesh_limits
)
  edge_terms = conservative_edge_terms(edge_diffusivity, edge_metrics, idx)
  stencil = neumann_stencil_2d(edge_terms, J, Δt, idx, mesh_limits)
  return stencil
end
