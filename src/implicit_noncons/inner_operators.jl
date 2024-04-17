
using StaticArrays, OffsetArrays

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

@kernel function inner_diffusion_op_kernel_2d!(
  A,
  b,
  α, # diffusivity
  u,
  source_term,
  Δt,
  cell_center_metrics,
  edge_metrics,
  grid_indices,
  matrix_indices,
  meanfunc::F,
  (ni, nj),
) where {F}
  idx = @index(Global, Linear)

  @inbounds begin
    grid_idx = grid_indices[idx]
    i, j = grid_idx.I

    metric_terms = non_cons_terms(cell_center_metrics, edge_metrics, grid_idx)

    uⁿᵢⱼ = u[i, j]
    sⁿᵢⱼ = source_term[i, j]
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

    stencil, rhs = inner_op_2d(metric_terms, diffusivity, uⁿᵢⱼ, Δt, sⁿᵢⱼ)

    mat_idx = matrix_indices[idx]
    #! format: off
    A[mat_idx, mat_idx - ni - 1] = stencil[-1, -1] # (i-1, j-1)
    A[mat_idx, mat_idx - ni]     = stencil[+0, -1] # (i  , j-1)
    A[mat_idx, mat_idx - ni + 1] = stencil[+1, -1] # (i+1, j-1)
    A[mat_idx, mat_idx - 1]      = stencil[-1, +0] # (i-1, j  )
    A[mat_idx, mat_idx]          = stencil[+0, +0] # (i  , j  )
    A[mat_idx, mat_idx + 1]      = stencil[+1, +0] # (i+1, j  )
    A[mat_idx, mat_idx + ni - 1] = stencil[-1, +1] # (i-1, j+1)
    A[mat_idx, mat_idx + ni]     = stencil[ 0, +1] # (i  , j+1)
    A[mat_idx, mat_idx + ni + 1] = stencil[+1, +1] # (i+1, j+1)
    #! format: on

    b[mat_idx] = rhs
  end
end

# ---------------------------------------------------------------------------
#  Operators
# ---------------------------------------------------------------------------

# # Generate a stencil for a single 1d cell in the interior
# @inline function _inner_diffusion_operator(
#   u::AbstractArray{T,1},
#   source_term::AbstractArray{T,1},
#   edge_diffusivity,
#   Δτ,
#   cell_center_metrics,
#   edge_metrics,
#   idx,
# ) where {T}

#   #
#   Jᵢ = cell_center_metrics.J[idx]
#   sᵢ = source_term[idx]
#   uᵢ = u[idx]

#   @unpack fᵢ₊½, fᵢ₋½ = conservative_edge_terms(edge_diffusivity, edge_metrics, idx)

#   #------------------------------------------------------------------------------
#   # Equations 3.43 and 3.44
#   #------------------------------------------------------------------------------
#   # fᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½^2) / Jᵢ₊½
#   # fᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½^2) / Jᵢ₋½

#   A = fᵢ₋½                       # (i-1)
#   B = -(fᵢ₋½ + fᵢ₊½ + Jᵢ / Δτ)  # (i)
#   C = fᵢ₊½                       # (i+1)
#   RHS = -(Jᵢ * sᵢ + uᵢ * Jᵢ / Δτ)

#   #------------------------------------------------------------------------------
#   # Assemble the stencil
#   #------------------------------------------------------------------------------
#   # Create a stencil matrix to hold the coefficients for u[i±1]

#   stencil = SVector{3,T}(A, B, C)

#   # use an offset so we can index via [+1, -1] for (i+1, j-1)
#   offset_stencil = OffsetVector(stencil, -1:1)

#   return offset_stencil, RHS
# end

# Generate a stencil for a single 2d cell in the interior
# @inline function _inner_diffusion_operator(
#   u::AbstractArray{T,2},
#   source_term::AbstractArray{T,2},
#   diffusivity::AbstractArray{T,2},
#   Δt,
#   cell_center_metrics,
#   edge_metrics,
#   idx::CartesianIndex{2},
#   meanfunc,
# ) where {T}

#   #

#   @unpack α, β, f_ξ², f_η², f_ξη = non_cons_terms(cell_center_metrics, edge_metrics, idx)

#   i, j = idx.I
#   aᵢⱼ = diffusivity[i, j]
#   aᵢ₊₁ⱼ = diffusivity[i + 1, j]
#   aᵢ₋₁ⱼ = diffusivity[i - 1, j]
#   aᵢⱼ₊₁ = diffusivity[i, j + 1]
#   aᵢⱼ₋₁ = diffusivity[i, j - 1]

#   aᵢ₊½ = meanfunc(aᵢⱼ, aᵢ₊₁ⱼ)
#   aᵢ₋½ = meanfunc(aᵢⱼ, aᵢ₋₁ⱼ)
#   aⱼ₊½ = meanfunc(aᵢⱼ, aᵢⱼ₊₁)
#   aⱼ₋½ = meanfunc(aᵢⱼ, aᵢⱼ₋₁)

#   inv_dt = inv(Δt)
#   uⁿ⁺¹ᵢⱼ = inv_dt + (
#     (aᵢ₊½ + aᵢ₋½) * f_ξ² + #
#     (aⱼ₊½ + aⱼ₋½) * f_η²   #
#   )

#   uⁿ⁺¹ᵢ₊₁ⱼ = (-aᵢⱼ * (α / 2) - aᵢ₊½ * f_ξ²)
#   uⁿ⁺¹ᵢ₋₁ⱼ = (+aᵢⱼ * (α / 2) - aᵢ₋½ * f_ξ²)
#   uⁿ⁺¹ᵢⱼ₊₁ = (-aᵢⱼ * (β / 2) - aⱼ₊½ * f_η²)
#   uⁿ⁺¹ᵢⱼ₋₁ = (+aᵢⱼ * (β / 2) - aⱼ₋½ * f_η²)
#   uⁿ⁺¹ᵢ₊₁ⱼ₋₁ = (aᵢ₊₁ⱼ + aᵢⱼ₋₁) * f_ξη
#   uⁿ⁺¹ᵢ₋₁ⱼ₊₁ = (aᵢ₋₁ⱼ + aᵢⱼ₊₁) * f_ξη
#   uⁿ⁺¹ᵢ₊₁ⱼ₊₁ = (-aᵢ₊₁ⱼ - aᵢⱼ₊₁) * f_ξη
#   uⁿ⁺¹ᵢ₋₁ⱼ₋₁ = (-aᵢ₋₁ⱼ - aᵢⱼ₋₁) * f_ξη

#   uⁿᵢⱼ = u[i, j]
#   rhs = source_term[i, j] + uⁿᵢⱼ * inv_dt

#   #! format: off
#   stencil = @SMatrix [
#     uⁿ⁺¹ᵢ₋₁ⱼ₋₁ uⁿ⁺¹ᵢⱼ₋₁ uⁿ⁺¹ᵢ₊₁ⱼ₋₁
#     uⁿ⁺¹ᵢ₋₁ⱼ   uⁿ⁺¹ᵢⱼ   uⁿ⁺¹ᵢ₊₁ⱼ
#     uⁿ⁺¹ᵢ₋₁ⱼ₊₁ uⁿ⁺¹ᵢⱼ₊₁ uⁿ⁺¹ᵢ₊₁ⱼ₊₁
#   ]
#   #! format: on

#   return OffsetMatrix(stencil, (-1:1, -1:1)), rhs
# end

function inner_op_2d(metric_terms, diffusivity, uⁿᵢⱼ::T, Δt::T, s::T) where {T}
  @unpack α, β, f_ξ², f_η², f_ξη = metric_terms
  @unpack aᵢⱼ, aᵢ₊₁ⱼ, aᵢ₋₁ⱼ, aᵢⱼ₊₁, aᵢⱼ₋₁, aᵢ₊½, aᵢ₋½, aⱼ₊½, aⱼ₋½ = diffusivity

  inv_dt = inv(Δt)
  uⁿ⁺¹ᵢⱼ = inv_dt + (
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

  rhs = s + uⁿᵢⱼ * inv_dt

  #! format: off
  stencil = @SMatrix [
    uⁿ⁺¹ᵢ₋₁ⱼ₋₁ uⁿ⁺¹ᵢⱼ₋₁ uⁿ⁺¹ᵢ₊₁ⱼ₋₁
    uⁿ⁺¹ᵢ₋₁ⱼ   uⁿ⁺¹ᵢⱼ   uⁿ⁺¹ᵢ₊₁ⱼ
    uⁿ⁺¹ᵢ₋₁ⱼ₊₁ uⁿ⁺¹ᵢⱼ₊₁ uⁿ⁺¹ᵢ₊₁ⱼ₊₁
  ]
  #! format: on

  return OffsetMatrix(stencil, (-1:1, -1:1)), rhs
end

# # Generate a stencil for a single 3d cell in the interior
# @inline function _inner_diffusion_operator(
#   u::AbstractArray{T,3},
#   source_term::AbstractArray{T,3},
#   edge_diffusivity,
#   Δτ,
#   cell_center_metrics,
#   edge_metrics,
#   idx,
# ) where {T}

#   #
#   Jᵢⱼₖ = cell_center_metrics.J[idx]
#   sᵢⱼₖ = source_term[idx]
#   uᵢⱼₖ = u[idx]

#   @unpack fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, fₖ₊½, fₖ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½, gₖ₊½, gₖ₋½ = conservative_edge_terms(
#     edge_diffusivity, edge_metrics, idx
#   )

#   # #------------------------------------------------------------------------------
#   # # Equations 3.43 and 3.44
#   # #------------------------------------------------------------------------------
#   # fᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
#   # fᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2) / Jᵢ₋½
#   # fⱼ₊½ = a_ⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½
#   # fⱼ₋½ = a_ⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2) / Jⱼ₋½

#   # #------------------------------------------------------------------------------
#   # # cross terms (Equations 3.45 and 3.46)
#   # #------------------------------------------------------------------------------
#   # gᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½ * Jηx_ᵢ₊½ + Jξy_ᵢ₊½ * Jηy_ᵢ₊½) / (4Jᵢ₊½)
#   # gᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½ * Jηx_ᵢ₋½ + Jξy_ᵢ₋½ * Jηy_ᵢ₋½) / (4Jᵢ₋½)
#   # gⱼ₊½ = a_ⱼ₊½ * (Jξx_ⱼ₊½ * Jηx_ⱼ₊½ + Jξy_ⱼ₊½ * Jηy_ⱼ₊½) / (4Jⱼ₊½)
#   # gⱼ₋½ = a_ⱼ₋½ * (Jξx_ⱼ₋½ * Jηx_ⱼ₋½ + Jξy_ⱼ₋½ * Jηy_ⱼ₋½) / (4Jⱼ₋½)

#   A = gᵢ₋½ + gⱼ₋½                              # (i-1,j-1)
#   B = fⱼ₋½ - gᵢ₊½ + gᵢ₋½                       # (i  ,j-1)
#   C = -gᵢ₊½ - gⱼ₋½                             # (i+1,j-1)
#   D = fᵢ₋½ - gⱼ₊½ + gⱼ₋½                       # (i-1,j)
#   F = fᵢ₊½ + gⱼ₊½ - gⱼ₋½                       # (i+1,j)
#   G = -gᵢ₋½ - gⱼ₊½                             # (i-1,j+1)
#   H = fⱼ₊½ + gᵢ₊½ - gᵢ₋½                       # (i  ,j+1)
#   I = gᵢ₊½ + gⱼ₊½                              # (i+1,j+1)
#   E = -(fᵢ₋½ + fⱼ₋½ + fᵢ₊½ + fⱼ₊½ + Jᵢⱼₖ / Δτ)  # (i,j)
#   RHS = -(Jᵢⱼₖ * sᵢⱼₖ + uᵢⱼₖ * Jᵢⱼₖ / Δτ)

#   #------------------------------------------------------------------------------
#   # Assemble the stencil
#   #------------------------------------------------------------------------------

#   # Create a stencil matrix to hold the coefficients for u[i±1,j±1]

#   # Don't use an offset matrix here b/c benchmarking showed it was
#   # ~3x slower. It makes the indexing here more of a pain,
#   # but 3x slower is a big deal for this performance critical section
#   stencil = SMatrix{3,3,T}(A, B, C, D, E, F, G, H, I)

#   # use an offset so we can index via [+1, -1, +1] for (i+1, j-1, k+1)
#   offset_stencil = OffsetMatrix(SMatrix{3,3}(stencil), -1:1, -1:1)

#   return offset_stencil, RHS
# end
