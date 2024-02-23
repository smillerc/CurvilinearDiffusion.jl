
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

  @inbounds begin
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

  len = ni * nj

  @inbounds begin
    grid_idx = grid_indices[idx]
    mat_idx = matrix_indices[idx]

    i, j = grid_idx.I
    edge_diffusivity = (
      αᵢ₊½=mean_func(α[i, j], α[i + 1, j]),
      αᵢ₋½=mean_func(α[i, j], α[i - 1, j]),
      αⱼ₊½=mean_func(α[i, j], α[i, j + 1]),
      αⱼ₋½=mean_func(α[i, j], α[i, j - 1]),
    )

    stencil, rhs = _neumann_boundary_diffusion_operator(
      u, source_term, edge_diffusivity, Δt, cell_center_metrics, edge_metrics, grid_idx, loc
    )

    A[mat_idx, mat_idx] = stencil[+0, +0] # (i, j)

    ip1 = true
    jp1 = true
    im1 = true
    jm1 = true
    if loc === :ilo
      im1 = false
    elseif loc === :ihi
      ip1 = false
    elseif loc === :jlo
      jm1 = false
    elseif loc === :jhi
      jp1 = false
    end

    #! format: off
    if ((im1 && jm1) && (1 <= (mat_idx - ni - 1 ) <= len)) A[mat_idx, mat_idx - ni - 1] = stencil[-1, -1] end # (i-1, j-1)
    if ((       jm1) && (1 <= (mat_idx - ni     ) <= len)) A[mat_idx, mat_idx - ni    ] = stencil[+0, -1] end # (i  , j-1)
    if ((im1       ) && (1 <= (mat_idx - 1      ) <= len)) A[mat_idx, mat_idx - 1     ] = stencil[-1, +0] end # (i-1, j  )
    if ((im1 && jp1) && (1 <= (mat_idx + ni - 1 ) <= len)) A[mat_idx, mat_idx + ni - 1] = stencil[-1, +1] end # (i-1, j+1)
    if ((       jp1) && (1 <= (mat_idx + ni     ) <= len)) A[mat_idx, mat_idx + ni    ] = stencil[ 0, +1] end # (i  , j+1)
    if ((ip1 && jm1) && (1 <= (mat_idx - ni + 1 ) <= len)) A[mat_idx, mat_idx - ni + 1] = stencil[+1, -1] end # (i+1, j-1)
    if ((ip1       ) && (1 <= (mat_idx + 1      ) <= len)) A[mat_idx, mat_idx + 1     ] = stencil[+1, +0] end # (i+1, j  )
    if ((ip1 && jp1) && (1 <= (mat_idx + ni + 1 ) <= len)) A[mat_idx, mat_idx + ni + 1] = stencil[+1, +1] end # (i+1, j+1)
    #! format: on

    b[mat_idx] = rhs
  end
end

# ---------------------------------------------------------------------------
#  Operators
# ---------------------------------------------------------------------------

# Generate a stencil for a 1D neumann boundary condition
@inline function _neumann_boundary_diffusion_operator(
  u::AbstractArray{T,2},
  source_term::AbstractArray{T,2},
  edge_diffusivity,
  Δτ,
  cell_center_metrics,
  edge_metrics,
  idx::CartesianIndex{2},
  loc::Symbol,
) where {T}

  #
  Jᵢ = cell_center_metrics.J[idx]
  sᵢ = source_term[idx]
  uᵢ = u[idx]

  @unpack fᵢ₊½, fᵢ₋½ = conservative_edge_terms(edge_diffusivity, edge_metrics, idx)

  if loc === :ilo
    fᵢ₋½ = zero(T)
  elseif loc === :ihi
    fᵢ₊½ = zero(T)
  end

  A = fᵢ₋½                     # (i-1,j)
  B = -(fᵢ₋½ + fᵢ₊½ + Jᵢ / Δτ) # (i,j)
  C = fᵢ₊½                     # (i+1,j)
  RHS = -(Jᵢ * sᵢ + uᵢ * Jᵢ / Δτ)

  #------------------------------------------------------------------------------
  # Assemble the stencil
  #------------------------------------------------------------------------------

  stencil = SVector{3,T}(A, B, C)

  # use an offset so we can index via [+1, -1] for (i+1, j-1)
  offset_stencil = OffsetVector(stencil, -1:1)

  return offset_stencil, RHS
end

# Generate a stencil for a 2D neumann boundary condition
@inline function _neumann_boundary_diffusion_operator(
  u::AbstractArray{T,2},
  source_term::AbstractArray{T,2},
  edge_diffusivity,
  Δτ,
  cell_center_metrics,
  edge_metrics,
  idx::CartesianIndex{2},
  loc::Symbol,
) where {T}

  #
  Jᵢⱼ = cell_center_metrics.J[idx]
  sᵢⱼ = source_term[idx]
  uᵢⱼ = u[idx]

  @unpack fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½ = conservative_edge_terms(
    edge_diffusivity, edge_metrics, idx
  )

  # Create a stencil matrix to hold the coefficients for u[i±1,j±1]
  if loc === :ilo
    fᵢ₋½ = zero(T)
    gᵢ₋½ = zero(T)
  elseif loc === :ihi
    fᵢ₊½ = zero(T)
    gᵢ₊½ = zero(T)
  elseif loc === :jlo
    fⱼ₋½ = zero(T)
    gⱼ₋½ = zero(T)
  elseif loc === :jhi
    fⱼ₊½ = zero(T)
    gⱼ₊½ = zero(T)
  end

  A = gᵢ₋½ + gⱼ₋½                              # (i-1,j-1)
  B = fⱼ₋½ - gᵢ₊½ + gᵢ₋½                       # (i  ,j-1)
  C = -gᵢ₊½ - gⱼ₋½                             # (i+1,j-1)
  D = fᵢ₋½ - gⱼ₊½ + gⱼ₋½                       # (i-1,j)
  F = fᵢ₊½ + gⱼ₊½ - gⱼ₋½                       # (i+1,j)
  G = -gᵢ₋½ - gⱼ₊½                             # (i-1,j+1)
  H = fⱼ₊½ + gᵢ₊½ - gᵢ₋½                       # (i  ,j+1)
  I = gᵢ₊½ + gⱼ₊½                              # (i+1,j+1)
  E = -(fᵢ₋½ + fⱼ₋½ + fᵢ₊½ + fⱼ₊½ + Jᵢⱼ / Δτ)  # (i,j)
  RHS = -(Jᵢⱼ * sᵢⱼ + uᵢⱼ * Jᵢⱼ / Δτ)

  #------------------------------------------------------------------------------
  # Assemble the stencil
  #------------------------------------------------------------------------------

  stencil = SMatrix{3,3,T}(A, B, C, D, E, F, G, H, I)

  # use an offset so we can index via [+1, -1] for (i+1, j-1)
  offset_stencil = OffsetMatrix(SMatrix{3,3}(stencil), -1:1, -1:1)

  return offset_stencil, RHS
end
