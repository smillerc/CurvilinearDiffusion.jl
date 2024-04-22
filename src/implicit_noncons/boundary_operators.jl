
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
  α,
  Δt,
  cell_center_metrics,
  edge_metrics,
  grid_indices,
  matrix_indices,
  meanfunc::F,
  (ni, nj),
  loc::Int,
) where {F}
  idx = @index(Global, Linear)

  len = ni * nj

  begin
    grid_idx = grid_indices[idx]
    mat_idx = matrix_indices[idx]
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

    stencil = _neumann_boundary_diffusion_operator(metric_terms, diffusivity, Δt, loc)

    A[mat_idx, mat_idx] = stencil[+0, +0] # (i, j)

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
    if ((im1 && jm1) && (1 <= (mat_idx - ni - 1 ) <= len)) A[mat_idx, mat_idx - ni - 1] = stencil[-1, -1] end # (i-1, j-1)
    if ((       jm1) && (1 <= (mat_idx - ni     ) <= len)) A[mat_idx, mat_idx - ni    ] = stencil[+0, -1] end # (i  , j-1)
    if ((im1       ) && (1 <= (mat_idx - 1      ) <= len)) A[mat_idx, mat_idx - 1     ] = stencil[-1, +0] end # (i-1, j  )
    if ((im1 && jp1) && (1 <= (mat_idx + ni - 1 ) <= len)) A[mat_idx, mat_idx + ni - 1] = stencil[-1, +1] end # (i-1, j+1)
    if ((       jp1) && (1 <= (mat_idx + ni     ) <= len)) A[mat_idx, mat_idx + ni    ] = stencil[ 0, +1] end # (i  , j+1)
    if ((ip1 && jm1) && (1 <= (mat_idx - ni + 1 ) <= len)) A[mat_idx, mat_idx - ni + 1] = stencil[+1, -1] end # (i+1, j-1)
    if ((ip1       ) && (1 <= (mat_idx + 1      ) <= len)) A[mat_idx, mat_idx + 1     ] = stencil[+1, +0] end # (i+1, j  )
    if ((ip1 && jp1) && (1 <= (mat_idx + ni + 1 ) <= len)) A[mat_idx, mat_idx + ni + 1] = stencil[+1, +1] end # (i+1, j+1)
    #! format: on

  end
end

# ---------------------------------------------------------------------------
#  Operators
# ---------------------------------------------------------------------------

# Generate a stencil for a 1D neumann boundary condition
@inline function _neumann_boundary_diffusion_operator(
  u::AbstractArray{T,1},
  source_term::AbstractArray{T,1},
  edge_diffusivity,
  Δτ,
  cell_center_metrics,
  edge_metrics,
  idx,
  loc,
) where {T}

  #
  Jᵢ = cell_center_metrics.J[idx]
  sᵢ = source_term[idx]
  uᵢ = u[idx]

  @unpack fᵢ₊½, fᵢ₋½ = conservative_edge_terms(edge_diffusivity, edge_metrics, idx)

  if loc == 1 # :ilo
    fᵢ₋½ = zero(T)
  elseif loc == 2 # :ihi
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
  metric_terms, diffusivity, Δt::T, loc
) where {T}
  @unpack α, β, f_ξ², f_η², f_ξη = metric_terms
  @unpack aᵢⱼ, aᵢ₊₁ⱼ, aᵢ₋₁ⱼ, aᵢⱼ₊₁, aᵢⱼ₋₁, aᵢ₊½, aᵢ₋½, aⱼ₊½, aⱼ₋½ = diffusivity

  # Create a stencil matrix to hold the coefficients for u[i±1,j±1]
  if loc == 1 # :ilo
    aᵢ₋½ = zero(T)
  elseif loc == 2 # :ihi
    aᵢ₊½ = zero(T)
  elseif loc == 3 # :jlo
    aⱼ₋½ = zero(T)
  elseif loc == 4 # :jhi
    aⱼ₊½ = zero(T)
  end

  uⁿ⁺¹ᵢⱼ = inv(Δt) + (
    (aᵢ₊½ + aᵢ₋½) * f_ξ² + #
    (aⱼ₊½ + aⱼ₋½) * f_η²   #
  )
  uⁿ⁺¹ᵢ₊₁ⱼ = (-aᵢⱼ * (α / 2) - aᵢ₊½ * f_ξ²)
  uⁿ⁺¹ᵢ₋₁ⱼ = (+aᵢⱼ * (α / 2) - aᵢ₋½ * f_ξ²)
  uⁿ⁺¹ᵢⱼ₊₁ = (-aᵢⱼ * (β / 2) - aⱼ₊½ * f_η²)
  uⁿ⁺¹ᵢⱼ₋₁ = (+aᵢⱼ * (β / 2) - aⱼ₋½ * f_η²)
  uⁿ⁺¹ᵢ₊₁ⱼ₋₁ = (aᵢ₊₁ⱼ + aᵢⱼ₋₁) * f_ξη
  uⁿ⁺¹ᵢ₋₁ⱼ₊₁ = (aᵢ₋₁ⱼ + aᵢⱼ₊₁) * f_ξη
  uⁿ⁺¹ᵢ₊₁ⱼ₊₁ = (-aᵢ₊₁ⱼ - aᵢⱼ₊₁) * f_ξη
  uⁿ⁺¹ᵢ₋₁ⱼ₋₁ = (-aᵢ₋₁ⱼ - aᵢⱼ₋₁) * f_ξη

  #! format: off
  stencil = @SMatrix [
    uⁿ⁺¹ᵢ₋₁ⱼ₋₁ uⁿ⁺¹ᵢⱼ₋₁ uⁿ⁺¹ᵢ₊₁ⱼ₋₁
    uⁿ⁺¹ᵢ₋₁ⱼ   uⁿ⁺¹ᵢⱼ   uⁿ⁺¹ᵢ₊₁ⱼ
    uⁿ⁺¹ᵢ₋₁ⱼ₊₁ uⁿ⁺¹ᵢⱼ₊₁ uⁿ⁺¹ᵢ₊₁ⱼ₊₁
  ]
  #! format: on

  return OffsetMatrix(stencil, (-1:1, -1:1))
end
