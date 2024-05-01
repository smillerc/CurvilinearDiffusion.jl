
# ---------------------------------------------------------------------------
#  Kernels
# ---------------------------------------------------------------------------

@kernel function inner_diffusion_op_kernel_1d!(
  A,
  b,
  @Const(α),
  @Const(u),
  @Const(source_term),
  @Const(Δt),
  @Const(cell_center_metrics),
  @Const(edge_metrics),
  @Const(grid_indices),
  @Const(matrix_indices),
  @Const(mean_func::F),
  @Const((ni, nj))
) where {F}
  idx = @index(Global, Linear)

  @inbounds begin
    grid_idx = grid_indices[idx]

    i, = grid_idx.I
    edge_diffusivity = (
      αᵢ₊½=mean_func(α[i, j], α[i + 1]), αᵢ₋½=mean_func(α[i, j], α[i - 1])
    )

    stencil, rhs = _inner_diffusion_operator(
      u, source_term, edge_diffusivity, Δt, cell_center_metrics, edge_metrics, grid_idx
    )

    mat_idx = matrix_indices[idx]
    #! format: off
    A[mat_idx, mat_idx - 1]      = stencil[-1] # (i-1, j  )
    A[mat_idx, mat_idx]          = stencil[+0] # (i  , j  )
    A[mat_idx, mat_idx + 1]      = stencil[+1] # (i+1, j  )
    #! format: on

    b[mat_idx] = rhs
  end
end

@kernel function inner_diffusion_op_kernel_2d!(
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
    grid_idx = grid_indices[idx]

    i, j = grid_idx.I
    edge_diffusivity = (
      αᵢ₊½=mean_func(α[i, j], α[i + 1, j]),
      αᵢ₋½=mean_func(α[i, j], α[i - 1, j]),
      αⱼ₊½=mean_func(α[i, j], α[i, j + 1]),
      αⱼ₋½=mean_func(α[i, j], α[i, j - 1]),
    )

    J = cell_center_metrics.J[grid_idx]
    stencil = _inner_diffusion_operator(edge_diffusivity, Δt, J, edge_metrics, grid_idx)

    row = matrix_indices[idx]

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
    grid_idx = grid_indices[idx]

    i, j, k = grid_idx.I
    edge_diffusivity = (
      αᵢ₊½=mean_func(α[i, j, k], α[i + 1, j, k]),
      αᵢ₋½=mean_func(α[i, j, k], α[i - 1, j, k]),
      αⱼ₊½=mean_func(α[i, j, k], α[i, j + 1, k]),
      αⱼ₋½=mean_func(α[i, j, k], α[i, j - 1, k]),
      αₖ₊½=mean_func(α[i, j, k], α[i, j, k + 1]),
      αₖ₋½=mean_func(α[i, j, k], α[i, j, k - 1]),
    )

    J = cell_center_metrics.J[grid_idx]
    stencil = _inner_diffusion_operator(edge_diffusivity, Δt, J, edge_metrics, grid_idx)

    row = matrix_indices[idx]

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

# @kernel function working_inner_diffusion_op_kernel_2d!(
#   A,
#   b,
#   α,
#   u,
#   source_term,
#   Δt,
#   cell_center_metrics,
#   edge_metrics,
#   grid_indices,
#   matrix_indices,
#   mean_func::F,
#   (ni, nj),
# ) where {F}
#   idx = @index(Global, Linear)

#   @inbounds begin
#     grid_idx = grid_indices[idx]

#     i, j = grid_idx.I
#     edge_diffusivity = (
#       αᵢ₊½=mean_func(α[i, j], α[i + 1, j]),
#       αᵢ₋½=mean_func(α[i, j], α[i - 1, j]),
#       αⱼ₊½=mean_func(α[i, j], α[i, j + 1]),
#       αⱼ₋½=mean_func(α[i, j], α[i, j - 1]),
#     )

#     stencil, rhs = _inner_diffusion_operator(
#       u, source_term, edge_diffusivity, Δt, cell_center_metrics, edge_metrics, grid_idx
#     )

#     mat_idx = matrix_indices[idx]
#     #! format: off
#     A[mat_idx, mat_idx - ni - 1] = stencil[-1, -1] # (i-1, j-1)
#     A[mat_idx, mat_idx - ni]     = stencil[+0, -1] # (i  , j-1)
#     A[mat_idx, mat_idx - ni + 1] = stencil[+1, -1] # (i+1, j-1)
#     A[mat_idx, mat_idx - 1]      = stencil[-1, +0] # (i-1, j  )
#     A[mat_idx, mat_idx]          = stencil[+0, +0] # (i  , j  )
#     A[mat_idx, mat_idx + 1]      = stencil[+1, +0] # (i+1, j  )
#     A[mat_idx, mat_idx + ni - 1] = stencil[-1, +1] # (i-1, j+1)
#     A[mat_idx, mat_idx + ni]     = stencil[ 0, +1] # (i  , j+1)
#     A[mat_idx, mat_idx + ni + 1] = stencil[+1, +1] # (i+1, j+1)
#     #! format: on

#     b[mat_idx] = rhs
#   end
# end

# ---------------------------------------------------------------------------
#  Operators
# ---------------------------------------------------------------------------

# Generate a stencil for a single 2d cell in the interior
@inline function _inner_diffusion_operator(
  edge_diffusivity, Δτ, J, edge_metrics, idx::CartesianIndex{1}
)
  edge_terms = conservative_edge_terms(edge_diffusivity, edge_metrics, idx)
  stencil = stencil_1d(edge_terms, J, Δτ)
  return stencil
end

@inline function _inner_diffusion_operator(
  edge_diffusivity, Δτ, J, edge_metrics, idx::CartesianIndex{2}
)
  edge_terms = conservative_edge_terms(edge_diffusivity, edge_metrics, idx)
  stencil = stencil_2d(edge_terms, J, Δτ)
  return stencil
end

@inline function _inner_diffusion_operator(
  edge_diffusivity, Δτ, J, edge_metrics, idx::CartesianIndex{3}
)
  edge_terms = conservative_edge_terms(edge_diffusivity, edge_metrics, idx)
  stencil = stencil_3d(edge_terms, J, Δτ)
  return stencil
end

# @inline function working_inner_diffusion_operator(
#   u::AbstractArray{T,2},
#   source_term::AbstractArray{T,2},
#   edge_diffusivity,
#   Δτ,
#   cell_center_metrics,
#   edge_metrics,
#   idx,
# ) where {T}

#   #
#   Jᵢⱼ = cell_center_metrics.J[idx]
#   sᵢⱼ = source_term[idx]
#   uᵢⱼ = u[idx]

#   @unpack fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½ = conservative_edge_terms(
#     edge_diffusivity, edge_metrics, idx
#   )

#   A = gᵢ₋½ + gⱼ₋½                              # (i-1,j-1)
#   B = fⱼ₋½ - gᵢ₊½ + gᵢ₋½                       # (i  ,j-1)
#   C = -gᵢ₊½ - gⱼ₋½                             # (i+1,j-1)
#   D = fᵢ₋½ - gⱼ₊½ + gⱼ₋½                       # (i-1,j)
#   F = fᵢ₊½ + gⱼ₊½ - gⱼ₋½                       # (i+1,j)
#   G = -gᵢ₋½ - gⱼ₊½                             # (i-1,j+1)
#   H = fⱼ₊½ + gᵢ₊½ - gᵢ₋½                       # (i  ,j+1)
#   I = gᵢ₊½ + gⱼ₊½                              # (i+1,j+1)
#   E = -(fᵢ₋½ + fⱼ₋½ + fᵢ₊½ + fⱼ₊½ + Jᵢⱼ / Δτ)  # (i,j)
#   RHS = -(Jᵢⱼ * sᵢⱼ + uᵢⱼ * Jᵢⱼ / Δτ)

#   #------------------------------------------------------------------------------
#   # Assemble the stencil
#   #------------------------------------------------------------------------------

#   # Create a stencil matrix to hold the coefficients for u[i±1,j±1]

#   # Don't use an offset matrix here b/c benchmarking showed it was
#   # ~3x slower. It makes the indexing here more of a pain,
#   # but 3x slower is a big deal for this performance critical section

#   stencil = SMatrix{3,3,T}(A, B, C, D, E, F, G, H, I)

#   # use an offset so we can index via [+1, -1] for (i+1, j-1)
#   offset_stencil = OffsetMatrix(SMatrix{3,3}(stencil), -1:1, -1:1)

#   return offset_stencil, RHS
# end
