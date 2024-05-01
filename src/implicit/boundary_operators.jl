
# ---------------------------------------------------------------------------
#  Kernels
# ---------------------------------------------------------------------------

@kernel function boundary_diffusion_op_kernel_1d!(
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
  @Const((ni, nj)),
  @Const(loc::Symbol)
) where {F}
  idx = @index(Global, Linear)

  begin
    grid_idx = grid_indices[idx]
    mat_idx = matrix_indices[idx]

    i, = grid_idx.I
    edge_diffusivity = (αᵢ₊½=mean_func(α[i], α[i + 1]), αᵢ₋½=mean_func(α[i], α[i - 1]))

    stencil, rhs = _neumann_boundary_diffusion_operator(
      u, source_term, edge_diffusivity, Δt, cell_center_metrics, edge_metrics, grid_idx, loc
    )

    A[mat_idx, mat_idx] = stencil[0] # (i)

    ip1 = true
    im1 = true

    if loc === :ilo
      im1 = false
    elseif loc === :ihi
      ip1 = false
    end

    if (im1 && (1 <= (mat_idx - 1) <= ni))
      A[mat_idx, mat_idx - 1] = stencil[-1] # i-1
    end

    if (ip1 && (1 <= (mat_idx + 1) <= ni))
      A[mat_idx, mat_idx + 1] = stencil[+1] # i+1
    end

    b[mat_idx] = rhs
  end
end

@kernel function boundary_diffusion_op_kernel_2d!(
  A,
  # b,
  α,
  # u,
  # source_term,
  Δt,
  cell_center_metrics,
  edge_metrics,
  grid_indices,
  matrix_indices,
  mean_func::F,
  stencil_col_lookup,
  loc::Int,
) where {F}
  idx = @index(Global, Linear)

  _, ncols = size(A)
  begin
    grid_idx = grid_indices[idx]
    row = matrix_indices[idx]

    i, j = grid_idx.I
    edge_diffusivity = (
      αᵢ₊½=mean_func(α[i, j], α[i + 1, j]),
      αᵢ₋½=mean_func(α[i, j], α[i - 1, j]),
      αⱼ₊½=mean_func(α[i, j], α[i, j + 1]),
      αⱼ₋½=mean_func(α[i, j], α[i, j - 1]),
    )

    stencil = _neumann_boundary_diffusion_operator(
      edge_diffusivity, Δt, cell_center_metrics, edge_metrics, grid_idx, loc
    )

    ip1 = true
    jp1 = true
    im1 = true
    jm1 = true
    if loc == 1 # :ilo
      im1 = false
    elseif loc == 2 # :ihi
      ip1 = false
    elseif loc == 3 # :jlo
      jm1 = false
    elseif loc == 4 # :jhi
      jp1 = false
    end

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

    A[row, colᵢⱼ] = stencil[5]  #[+0, +0] # (i  , j  )

    if ((im1 && jm1) && (1 <= colᵢ₋₁ⱼ₋₁ <= ncols)) A[row, colᵢ₋₁ⱼ₋₁] = stencil[1] end # (i-1, j-1)
    if ((       jm1) && (1 <= colᵢⱼ₋₁   <= ncols)) A[row, colᵢⱼ₋₁  ] = stencil[2] end # (i  , j-1)
    if ((im1       ) && (1 <= colᵢ₊₁ⱼ₋₁ <= ncols)) A[row, colᵢ₊₁ⱼ₋₁] = stencil[3] end # (i+1, j-1)
    if ((im1 && jp1) && (1 <= colᵢ₋₁ⱼ   <= ncols)) A[row, colᵢ₋₁ⱼ  ] = stencil[4] end # (i-1, j  )
    if ((       jp1) && (1 <= colᵢ₊₁ⱼ   <= ncols)) A[row, colᵢ₊₁ⱼ  ] = stencil[6] end # (i+1, j  )
    if ((ip1 && jm1) && (1 <= colᵢ₋₁ⱼ₊₁ <= ncols)) A[row, colᵢ₋₁ⱼ₊₁] = stencil[7] end # (i-1, j+1)
    if ((ip1       ) && (1 <= colᵢⱼ₊₁   <= ncols)) A[row, colᵢⱼ₊₁  ] = stencil[8] end # (i  , j+1)
    if ((ip1 && jp1) && (1 <= colᵢ₊₁ⱼ₊₁ <= ncols)) A[row, colᵢ₊₁ⱼ₊₁] = stencil[9] end # (i+1, j+1)
    #! format: on

  end
end

@kernel function boundary_diffusion_op_kernel_3d!(
  A,
  α,
  Δt,
  cell_center_metrics,
  edge_metrics,
  grid_indices,
  matrix_indices,
  mean_func::F,
  stencil_col_lookup,
  loc::Int,
) where {F}
  idx = @index(Global, Linear)

  _, ncols = size(A)
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

    ip1 = jp1 = kp1 = im1 = jm1 = km1 = true

    if loc == 1 # :ilo
      im1 = false
    elseif loc == 2 # :ihi
      ip1 = false
    elseif loc == 3 # :jlo
      jm1 = false
    elseif loc == 4 # :jhi
      jp1 = false
    elseif loc == 5 # :klo
      km1 = false
    elseif loc == 6 # :khi
      kp1 = false
    end

    #! format: off
    colᵢ₋₁ⱼ₋₁ₖ₋₁ = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₋₁ₖ₋₁)
    colᵢⱼ₋₁ₖ₋₁   = row + first(stencil_col_lookup.ᵢⱼ₋₁ₖ₋₁)
    colᵢ₊₁ⱼ₋₁ₖ₋₁ = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₋₁ₖ₋₁)
    colᵢ₋₁ⱼₖ₋₁   = row + first(stencil_col_lookup.ᵢ₋₁ⱼₖ₋₁)
    colᵢⱼₖ₋₁     = row + first(stencil_col_lookup.ᵢⱼₖ₋₁)
    colᵢ₊₁ⱼₖ₋₁   = row + first(stencil_col_lookup.ᵢ₊₁ⱼₖ₋₁)
    colᵢ₋₁ⱼ₊₁ₖ₋₁ = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₊₁ₖ₋₁)
    colᵢⱼ₊₁ₖ₋₁   = row + first(stencil_col_lookup.ᵢⱼ₊₁ₖ₋₁)
    colᵢ₊₁ⱼ₊₁ₖ₋₁ = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₊₁ₖ₋₁)
    colᵢ₋₁ⱼ₋₁ₖ   = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₋₁ₖ)
    colᵢⱼ₋₁ₖ     = row + first(stencil_col_lookup.ᵢⱼ₋₁ₖ)
    colᵢ₊₁ⱼ₋₁ₖ   = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₋₁ₖ)
    colᵢ₋₁ⱼₖ     = row + first(stencil_col_lookup.ᵢ₋₁ⱼₖ)
    colᵢⱼₖ       = row + first(stencil_col_lookup.ᵢⱼₖ)
    colᵢ₊₁ⱼₖ     = row + first(stencil_col_lookup.ᵢ₊₁ⱼₖ)
    colᵢ₋₁ⱼ₊₁ₖ   = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₊₁ₖ)
    colᵢⱼ₊₁ₖ     = row + first(stencil_col_lookup.ᵢⱼ₊₁ₖ)
    colᵢ₊₁ⱼ₊₁ₖ   = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₊₁ₖ)
    colᵢ₋₁ⱼ₋₁ₖ₊₁ = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₋₁ₖ₊₁)
    colᵢⱼ₋₁ₖ₊₁   = row + first(stencil_col_lookup.ᵢⱼ₋₁ₖ₊₁)
    colᵢ₊₁ⱼ₋₁ₖ₊₁ = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₋₁ₖ₊₁)
    colᵢ₋₁ⱼₖ₊₁   = row + first(stencil_col_lookup.ᵢ₋₁ⱼₖ₊₁)
    colᵢⱼₖ₊₁     = row + first(stencil_col_lookup.ᵢⱼₖ₊₁)
    colᵢ₊₁ⱼₖ₊₁   = row + first(stencil_col_lookup.ᵢ₊₁ⱼₖ₊₁)
    colᵢ₋₁ⱼ₊₁ₖ₊₁ = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₊₁ₖ₊₁)
    colᵢⱼ₊₁ₖ₊₁   = row + first(stencil_col_lookup.ᵢⱼ₊₁ₖ₊₁)
    colᵢ₊₁ⱼ₊₁ₖ₊₁ = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₊₁ₖ₊₁)

    A[row, colᵢⱼₖ] = stencil[14]

    if ((im1 && jm1 && km1) && (1 <= colᵢ₋₁ⱼ₋₁ₖ₋₁ <= ncols))    A[row, colᵢ₋₁ⱼ₋₁ₖ₋₁ ]  = stencil[1] end
    if ((       jm1 && km1) && (1 <= colᵢⱼ₋₁ₖ₋₁ <= ncols))      A[row, colᵢⱼ₋₁ₖ₋₁   ]  = stencil[2] end
    if ((ip1 && jm1 && km1) && (1 <= colᵢ₊₁ⱼ₋₁ₖ₋₁ <= ncols))    A[row, colᵢ₊₁ⱼ₋₁ₖ₋₁ ]  = stencil[3] end
    if ((im1        && km1) && (1 <= colᵢ₋₁ⱼₖ₋₁ <= ncols))      A[row, colᵢ₋₁ⱼₖ₋₁   ]  = stencil[4] end
    if ((              km1) && (1 <= colᵢⱼₖ₋₁ <= ncols))        A[row, colᵢⱼₖ₋₁     ]  = stencil[5] end
    if ((ip1        && km1) && (1 <= colᵢ₊₁ⱼₖ₋₁ <= ncols))      A[row, colᵢ₊₁ⱼₖ₋₁   ]  = stencil[6] end
    if ((im1 && jp1 && km1) && (1 <= colᵢ₋₁ⱼ₊₁ₖ₋₁ <= ncols))    A[row, colᵢ₋₁ⱼ₊₁ₖ₋₁ ]  = stencil[7] end
    if ((       jp1 && km1) && (1 <= colᵢⱼ₊₁ₖ₋₁ <= ncols))      A[row, colᵢⱼ₊₁ₖ₋₁   ]  = stencil[8] end
    if ((ip1 && jp1 && km1) && (1 <= colᵢ₊₁ⱼ₊₁ₖ₋₁ <= ncols))    A[row, colᵢ₊₁ⱼ₊₁ₖ₋₁ ]  = stencil[9] end
    if ((im1 && jm1       ) && (1 <= colᵢ₋₁ⱼ₋₁ₖ <= ncols))      A[row, colᵢ₋₁ⱼ₋₁ₖ   ]  = stencil[10] end
    if ((       jm1       ) && (1 <= colᵢⱼ₋₁ₖ <= ncols))        A[row, colᵢⱼ₋₁ₖ     ]  = stencil[11] end
    if ((ip1 && jm1       ) && (1 <= colᵢ₊₁ⱼ₋₁ₖ <= ncols))      A[row, colᵢ₊₁ⱼ₋₁ₖ   ]  = stencil[12] end
    if ((im1              ) && (1 <= colᵢ₋₁ⱼₖ <= ncols))        A[row, colᵢ₋₁ⱼₖ     ]  = stencil[13] end
 
    if ((ip1              ) && (1 <= colᵢ₊₁ⱼₖ <= ncols))        A[row, colᵢ₊₁ⱼₖ     ]  = stencil[15] end
    if ((im1 && jp1       ) && (1 <= colᵢ₋₁ⱼ₊₁ₖ <= ncols))      A[row, colᵢ₋₁ⱼ₊₁ₖ   ]  = stencil[16] end
    if ((       jp1       ) && (1 <= colᵢⱼ₊₁ₖ <= ncols))        A[row, colᵢⱼ₊₁ₖ     ]  = stencil[17] end
    if ((ip1 && jp1       ) && (1 <= colᵢ₊₁ⱼ₊₁ₖ <= ncols))      A[row, colᵢ₊₁ⱼ₊₁ₖ   ]  = stencil[18] end
    if ((im1 && jm1 && kp1) && (1 <= colᵢ₋₁ⱼ₋₁ₖ₊₁ <= ncols))    A[row, colᵢ₋₁ⱼ₋₁ₖ₊₁ ]  = stencil[19] end
    if ((       jm1 && kp1) && (1 <= colᵢⱼ₋₁ₖ₊₁ <= ncols))      A[row, colᵢⱼ₋₁ₖ₊₁   ]  = stencil[20] end
    if ((ip1 && jm1 && kp1) && (1 <= colᵢ₊₁ⱼ₋₁ₖ₊₁ <= ncols))    A[row, colᵢ₊₁ⱼ₋₁ₖ₊₁ ]  = stencil[21] end
    if ((im1        && kp1) && (1 <= colᵢ₋₁ⱼₖ₊₁ <= ncols))      A[row, colᵢ₋₁ⱼₖ₊₁   ]  = stencil[22] end
    if ((              kp1) && (1 <= colᵢⱼₖ₊₁ <= ncols))        A[row, colᵢⱼₖ₊₁     ]  = stencil[23] end
    if ((ip1        && kp1) && (1 <= colᵢ₊₁ⱼₖ₊₁ <= ncols))      A[row, colᵢ₊₁ⱼₖ₊₁   ]  = stencil[24] end
    if ((im1 && jp1 && kp1) && (1 <= colᵢ₋₁ⱼ₊₁ₖ₊₁ <= ncols))    A[row, colᵢ₋₁ⱼ₊₁ₖ₊₁ ]  = stencil[25] end
    if ((       jp1 && kp1) && (1 <= colᵢⱼ₊₁ₖ₊₁ <= ncols))      A[row, colᵢⱼ₊₁ₖ₊₁   ]  = stencil[26] end
    if ((ip1 && jp1 && kp1) && (1 <= colᵢ₊₁ⱼ₊₁ₖ₊₁ <= ncols))    A[row, colᵢ₊₁ⱼ₊₁ₖ₊₁ ]  = stencil[27] end

    #! format: on

  end
end

# @kernel function boundary_diffusion_op_kernel_2d!(
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
#   loc::Int,
# ) where {F}
#   idx = @index(Global, Linear)

#   len = ni * nj

#   begin
#     grid_idx = grid_indices[idx]
#     mat_idx = matrix_indices[idx]

#     i, j = grid_idx.I
#     edge_diffusivity = (
#       αᵢ₊½=mean_func(α[i, j], α[i + 1, j]),
#       αᵢ₋½=mean_func(α[i, j], α[i - 1, j]),
#       αⱼ₊½=mean_func(α[i, j], α[i, j + 1]),
#       αⱼ₋½=mean_func(α[i, j], α[i, j - 1]),
#     )

#     stencil, rhs = _neumann_boundary_diffusion_operator(
#       u, source_term, edge_diffusivity, Δt, cell_center_metrics, edge_metrics, grid_idx, loc
#     )

#     A[mat_idx, mat_idx] = stencil[+0, +0] # (i, j)

#     ip1 = true
#     jp1 = true
#     im1 = true
#     jm1 = true
#     if loc == 1 # :ilo
#       im1 = false
#     elseif loc == 2 # :ihi
#       ip1 = false
#     elseif loc == 3 # :jlo
#       jm1 = false
#     elseif loc == 4 # :jhi
#       jp1 = false
#     end

#     #! format: off
#     if ((im1 && jm1) && (1 <= (mat_idx - ni - 1 ) <= len)) A[mat_idx, mat_idx - ni - 1] = stencil[-1, -1] end # (i-1, j-1)
#     if ((       jm1) && (1 <= (mat_idx - ni     ) <= len)) A[mat_idx, mat_idx - ni    ] = stencil[+0, -1] end # (i  , j-1)
#     if ((im1       ) && (1 <= (mat_idx - 1      ) <= len)) A[mat_idx, mat_idx - 1     ] = stencil[-1, +0] end # (i-1, j  )
#     if ((im1 && jp1) && (1 <= (mat_idx + ni - 1 ) <= len)) A[mat_idx, mat_idx + ni - 1] = stencil[-1, +1] end # (i-1, j+1)
#     if ((       jp1) && (1 <= (mat_idx + ni     ) <= len)) A[mat_idx, mat_idx + ni    ] = stencil[ 0, +1] end # (i  , j+1)
#     if ((ip1 && jm1) && (1 <= (mat_idx - ni + 1 ) <= len)) A[mat_idx, mat_idx - ni + 1] = stencil[+1, -1] end # (i+1, j-1)
#     if ((ip1       ) && (1 <= (mat_idx + 1      ) <= len)) A[mat_idx, mat_idx + 1     ] = stencil[+1, +0] end # (i+1, j  )
#     if ((ip1 && jp1) && (1 <= (mat_idx + ni + 1 ) <= len)) A[mat_idx, mat_idx + ni + 1] = stencil[+1, +1] end # (i+1, j+1)
#     #! format: on

#     b[mat_idx] = rhs
#   end
# end

# ---------------------------------------------------------------------------
#  Operators
# ---------------------------------------------------------------------------

# Generate a stencil for a 2D neumann boundary condition
@inline function _neumann_boundary_diffusion_operator(
  edge_diffusivity, Δτ, J, edge_metrics, idx::CartesianIndex{2}, loc
)
  T = eltype(edge_diffusivity)

  @unpack a_Jξ²ᵢ₊½, a_Jξ²ᵢ₋½, a_Jη²ⱼ₊½, a_Jη²ⱼ₋½, a_Jξηᵢ₊½, a_Jξηᵢ₋½, a_Jηξⱼ₊½, a_Jηξⱼ₋½ = conservative_edge_terms(
    edge_diffusivity, edge_metrics, idx
  )

  # Create a stencil matrix to hold the coefficients for u[i±1,j±1]
  if loc == 1 # :ilo
    a_Jξ²ᵢ₋½ = zero(T)
    a_Jξηᵢ₋½ = zero(T)
  elseif loc == 2 # :ihi
    a_Jξ²ᵢ₊½ = zero(T)
    a_Jξηᵢ₊½ = zero(T)
  elseif loc == 3 # :jlo
    a_Jη²ⱼ₋½ = zero(T)
    a_Jηξⱼ₋½ = zero(T)
  elseif loc == 4 # :jhi
    a_Jη²ⱼ₊½ = zero(T)
    a_Jηξⱼ₊½ = zero(T)
  else
    error("bad boundary location")
  end

  edge_terms = (;
    a_Jξ²ᵢ₊½, a_Jξ²ᵢ₋½, a_Jη²ⱼ₊½, a_Jη²ⱼ₋½, a_Jξηᵢ₊½, a_Jξηᵢ₋½, a_Jηξⱼ₊½, a_Jηξⱼ₋½
  )
  stencil = stencil_2d(edge_terms, J, Δτ)

  return stencil
end

# Generate a stencil for a 3D neumann boundary condition
@inline function _neumann_boundary_diffusion_operator(
  edge_diffusivity, Δτ, J, edge_metrics, idx::CartesianIndex{3}, loc
)
  T = eltype(edge_diffusivity)

  edge_terms = conservative_edge_terms(edge_diffusivity, edge_metrics, idx)

  a_Jξ²ᵢ₊½ = edge_terms.a_Jξ²ᵢ₊½
  a_Jξηᵢ₊½ = edge_terms.a_Jξηᵢ₊½
  a_Jξζᵢ₊½ = edge_terms.a_Jξζᵢ₊½

  a_Jξ²ᵢ₋½ = edge_terms.a_Jξ²ᵢ₋½
  a_Jξηᵢ₋½ = edge_terms.a_Jξηᵢ₋½
  a_Jξζᵢ₋½ = edge_terms.a_Jξζᵢ₋½

  a_Jηξⱼ₊½ = edge_terms.a_Jηξⱼ₊½
  a_Jη²ⱼ₊½ = edge_terms.a_Jη²ⱼ₊½
  a_Jηζⱼ₊½ = edge_terms.a_Jηζⱼ₊½

  a_Jηξⱼ₋½ = edge_terms.a_Jηξⱼ₋½
  a_Jη²ⱼ₋½ = edge_terms.a_Jη²ⱼ₋½
  a_Jηζⱼ₋½ = edge_terms.a_Jηζⱼ₋½

  a_Jζξₖ₊½ = edge_terms.a_Jζξₖ₊½
  a_Jζηₖ₊½ = edge_terms.a_Jζηₖ₊½
  a_Jζ²ₖ₊½ = edge_terms.a_Jζ²ₖ₊½

  a_Jζξₖ₋½ = edge_terms.a_Jζξₖ₋½
  a_Jζηₖ₋½ = edge_terms.a_Jζηₖ₋½
  a_Jζ²ₖ₋½ = edge_terms.a_Jζ²ₖ₋½

  # Create a stencil matrix to hold the coefficients for u[i±1,j±1]
  if loc == 1 # :ilo
    a_Jξ²ᵢ₋½ = a_Jξηᵢ₋½ = a_Jξζᵢ₋½ = zero(T)

  elseif loc == 2 # :ihi
    a_Jξ²ᵢ₊½ = a_Jξηᵢ₊½ = a_Jξζᵢ₊½ = zero(T)

  elseif loc == 3 # :jlo
    a_Jηξⱼ₋½ = a_Jη²ⱼ₋½ = a_Jηζⱼ₋½ = zero(T)

  elseif loc == 4 # :jhi
    a_Jηξⱼ₊½ = a_Jη²ⱼ₊½ = a_Jηζⱼ₊½ = zero(T)

  elseif loc == 5 # :klo
    a_Jζξₖ₋½ = a_Jζηₖ₋½ = a_Jζ²ₖ₋½ = zero(T)

  elseif loc == 6 # :khi
    a_Jζξₖ₊½ = a_Jζηₖ₊½ = a_Jζ²ₖ₊½ = zero(T)
  else
    error("bad boundary location")
  end

  edge_terms = (;
    a_Jξ²ᵢ₊½,
    a_Jξ²ᵢ₋½,
    a_Jξηᵢ₊½,
    a_Jξηᵢ₋½,
    a_Jξζᵢ₊½,
    a_Jξζᵢ₋½,
    a_Jηξⱼ₊½,
    a_Jηξⱼ₋½,
    a_Jη²ⱼ₊½,
    a_Jη²ⱼ₋½,
    a_Jηζⱼ₊½,
    a_Jηζⱼ₋½,
    a_Jζξₖ₊½,
    a_Jζξₖ₋½,
    a_Jζηₖ₊½,
    a_Jζηₖ₋½,
    a_Jζ²ₖ₊½,
    a_Jζ²ₖ₋½,
  )

  stencil = stencil_3d(edge_terms, J, Δτ)

  return stencil
end