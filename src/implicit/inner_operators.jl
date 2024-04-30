
using StaticArrays, OffsetArrays

const ilo_loc = 1
const ihi_loc = 2
const jlo_loc = 3
const jhi_loc = 4
const klo_loc = 5
const khi_loc = 6

const ilojlo_loc = 7
const ilojhi_loc = 8
const ihijlo_loc = 9
const ihijhi_loc = 10

const ilojloklo_loc = 11
const ilojhiklo_loc = 12
const ihijloklo_loc = 13
const ihijhiklo_loc = 14

const ilojlokhi_loc = 15
const ilojhikhi_loc = 16
const ihijlokhi_loc = 17
const ihijhikhi_loc = 18

function lame_o()
  ∂u∂t =
    f_ξ² * (aᵢ₊½ * (uⁿ⁺¹ᵢ₊₁ⱼ - uⁿ⁺¹ᵢⱼ) - aᵢ₋½ * (uⁿ⁺¹ᵢⱼ - uⁿ⁺¹ᵢ₋₁ⱼ)) +
    f_η² * (aⱼ₊½ * (uⁿ⁺¹ᵢⱼ₊₁ - uⁿ⁺¹ᵢⱼ) - aⱼ₋½ * (uⁿ⁺¹ᵢⱼ - uⁿ⁺¹ᵢⱼ₋₁)) +
    f_ξη * (
      aᵢ₊₁ⱼ * (uⁿ⁺¹ᵢ₊₁ⱼ₊₁ - uⁿ⁺¹ᵢ₊₁ⱼ₋₁) - # ∂u/∂η
      aᵢ₋₁ⱼ * (uⁿ⁺¹ᵢ₋₁ⱼ₊₁ - uⁿ⁺¹ᵢ₋₁ⱼ₋₁)   # ∂u/∂η
    ) + # ∂/∂ξ
    f_ξη * (
      aᵢⱼ₊₁ * (uⁿ⁺¹ᵢ₊₁ⱼ₊₁ - uⁿ⁺¹ᵢ₋₁ⱼ₊₁) - # ∂u/∂ξ
      aᵢⱼ₋₁ * (uⁿ⁺¹ᵢ₊₁ⱼ₋₁ - uⁿ⁺¹ᵢ₋₁ⱼ₋₁)   # ∂u/∂ξ
    ) + # ∂/∂η
    aᵢⱼ * 0.5α * (uⁿ⁺¹ᵢ₊₁ⱼ - uⁿ⁺¹ᵢ₋₁ⱼ) +
    aᵢⱼ * 0.5β * (uⁿ⁺¹ᵢⱼ₊₁ - uⁿ⁺¹ᵢⱼ₋₁) +
    s

  coeffs = (
    uⁿ⁺¹ᵢ₊₁ⱼ=f_ξ² * aᵢ₊½ + aᵢⱼ * 0.5α, #
  )
end

# ---------------------------------------------------------------------------
#  Kernels
# ---------------------------------------------------------------------------
# @kernel function inner_diffusion_op_kernel_2d!(
#   A,
#   α,
#   Δt,
#   cell_center_metrics,
#   edge_metrics,
#   grid_indices,
#   matrix_indices,
#   meanfunc::F,
#   stencil_col_lookup,
# ) where {F}
#   idx = @index(Global, Linear)

#   begin
#     grid_idx = grid_indices[idx]
#     row = matrix_indices[idx]
#     i, j = grid_idx.I

#     metric_terms = non_cons_terms(cell_center_metrics, edge_metrics, grid_idx)

#     aᵢⱼ = α[i, j]
#     aᵢ₊₁ⱼ = α[i + 1, j]
#     aᵢ₋₁ⱼ = α[i - 1, j]
#     aᵢⱼ₊₁ = α[i, j + 1]
#     aᵢⱼ₋₁ = α[i, j - 1]

#     diffusivity = (;
#       aᵢⱼ,
#       aᵢ₊₁ⱼ,
#       aᵢ₋₁ⱼ,
#       aᵢⱼ₊₁,
#       aᵢⱼ₋₁,
#       aᵢ₊½=meanfunc(aᵢⱼ, aᵢ₊₁ⱼ),
#       aᵢ₋½=meanfunc(aᵢⱼ, aᵢ₋₁ⱼ),
#       aⱼ₊½=meanfunc(aᵢⱼ, aᵢⱼ₊₁),
#       aⱼ₋½=meanfunc(aᵢⱼ, aᵢⱼ₋₁),
#     )

#     #! format: off
#     colᵢ₋₁ⱼ₋₁ = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₋₁)
#     colᵢⱼ₋₁ =   row + first(stencil_col_lookup.ᵢⱼ₋₁)
#     colᵢ₊₁ⱼ₋₁ = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₋₁)
#     colᵢ₋₁ⱼ =   row + first(stencil_col_lookup.ᵢ₋₁ⱼ)
#     colᵢⱼ =     row + first(stencil_col_lookup.ᵢⱼ)
#     colᵢ₊₁ⱼ =   row + first(stencil_col_lookup.ᵢ₊₁ⱼ)
#     colᵢ₋₁ⱼ₊₁ = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₊₁)
#     colᵢⱼ₊₁ =   row + first(stencil_col_lookup.ᵢⱼ₊₁)
#     colᵢ₊₁ⱼ₊₁ = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₊₁)

#     stencil = inner_op_2d(metric_terms, diffusivity, Δt)

#     if i == 8 && j == 8
#       @show grid_indices
#       @show stencil 
#       @show row
#       @show colᵢ₋₁ⱼ₋₁
#       @show colᵢⱼ₋₁
#       @show colᵢ₊₁ⱼ₋₁
#       @show colᵢ₋₁ⱼ
#       @show colᵢⱼ
#       @show colᵢ₊₁ⱼ
#       @show colᵢ₋₁ⱼ₊₁
#       @show colᵢⱼ₊₁
#       @show colᵢ₊₁ⱼ₊₁
#     end

#     A[row, colᵢ₋₁ⱼ₋₁] = stencil[1]  #[-1, -1] # (i-1, j-1)
#     A[row, colᵢⱼ₋₁  ] = stencil[2]  #[+0, -1] # (i  , j-1)
#     A[row, colᵢ₊₁ⱼ₋₁] = stencil[3]  #[+1, -1] # (i+1, j-1)
#     A[row, colᵢ₋₁ⱼ  ] = stencil[4]  #[-1, +0] # (i-1, j  )
#     A[row, colᵢⱼ    ] = stencil[5]  #[+0, +0] # (i  , j  )
#     A[row, colᵢ₊₁ⱼ  ] = stencil[6]  #[+1, +0] # (i+1, j  )
#     A[row, colᵢ₋₁ⱼ₊₁] = stencil[7]  #[-1, +1] # (i-1, j+1)
#     A[row, colᵢⱼ₊₁  ] = stencil[8]  #[ 0, +1] # (i  , j+1)
#     A[row, colᵢ₊₁ⱼ₊₁] = stencil[9]  #[+1, +1] # (i+1, j+1)
#     #! format: on

#   end
# end

@kernel function inner_diffusion_op_kernel_2d!(
  A,
  α,
  Δt,
  cell_center_metrics,
  edge_metrics,
  grid_indices,
  matrix_indices,
  meanfunc::F,
  (ni, nj),
) where {F}
  idx = @index(Global, Linear)

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

    stencil = inner_op_2d(metric_terms, diffusivity, Δt)

    #! format: off
    A[mat_idx, mat_idx - ni - 1] = stencil[1] #[-1, -1] # (i-1, j-1)
    A[mat_idx, mat_idx - ni]     = stencil[2] #[+0, -1] # (i  , j-1)
    A[mat_idx, mat_idx - ni + 1] = stencil[3] #[+1, -1] # (i+1, j-1)
    A[mat_idx, mat_idx - 1]      = stencil[4] #[-1, +0] # (i-1, j  )
    A[mat_idx, mat_idx]          = stencil[5] #[+0, +0] # (i  , j  )
    A[mat_idx, mat_idx + 1]      = stencil[6] #[+1, +0] # (i+1, j  )
    A[mat_idx, mat_idx + ni - 1] = stencil[7] #[-1, +1] # (i-1, j+1)
    A[mat_idx, mat_idx + ni]     = stencil[8] #[ 0, +1] # (i  , j+1)
    A[mat_idx, mat_idx + ni + 1] = stencil[9] #[+1, +1] # (i+1, j+1)
    #! format: on

    # b[mat_idx] = rhs
  end
end

# ---------------------------------------------------------------------------
#  Operators
# ---------------------------------------------------------------------------

function inner_op_1d(metric_terms, diffusivity, Δt::T) where {T}

  #
  @unpack α, f_ξ² = metric_terms
  @unpack aᵢ, aᵢ₊₁, aᵢ₋₁, aᵢ₊½, aᵢ₋½ = diffusivity

  # current cell
  uⁿ⁺¹ᵢ = one(T) + ((aᵢ₊½ + aᵢ₋½) * f_ξ²) * Δt

  # cardinal terms
  uⁿ⁺¹ᵢ₊₁ = (-aᵢ₊½ * f_ξ² - aᵢ * (α / 2)) * Δt
  uⁿ⁺¹ᵢ₋₁ = (-aᵢ₋½ * f_ξ² + aᵢ * (α / 2)) * Δt

  stencil = SVector{3,T}(uⁿ⁺¹ᵢ₋₁, uⁿ⁺¹ᵢ, uⁿ⁺¹ᵢ₊₁)

  return stencil
end

function inner_op_2d(metric_terms, diffusivity, Δt::T) where {T}

  #
  @unpack α, β, f_ξ², f_η², f_ξη = metric_terms
  @unpack aᵢⱼ, aᵢ₊₁ⱼ, aᵢ₋₁ⱼ, aᵢⱼ₊₁, aᵢⱼ₋₁, aᵢ₊½, aᵢ₋½, aⱼ₊½, aⱼ₋½ = diffusivity


  # current cell
  uⁿ⁺¹ᵢⱼ = one(T) + ((aᵢ₊½ + aᵢ₋½) * f_ξ² + (aⱼ₊½ + aⱼ₋½) * f_η²) * Δt

  # cardinal terms
  uⁿ⁺¹ᵢ₊₁ⱼ = (-aᵢ₊½ * f_ξ² - aᵢⱼ * (α / 2)) * Δt
  uⁿ⁺¹ᵢ₋₁ⱼ = (-aᵢ₋½ * f_ξ² + aᵢⱼ * (α / 2)) * Δt
  uⁿ⁺¹ᵢⱼ₊₁ = (-aⱼ₊½ * f_η² - aᵢⱼ * (β / 2)) * Δt
  uⁿ⁺¹ᵢⱼ₋₁ = (-aⱼ₋½ * f_η² + aᵢⱼ * (β / 2)) * Δt

  # corner terms
  uⁿ⁺¹ᵢ₊₁ⱼ₋₁ = (+aᵢ₊₁ⱼ + aᵢⱼ₋₁) * f_ξη * Δt
  uⁿ⁺¹ᵢ₊₁ⱼ₊₁ = (-aᵢ₊₁ⱼ - aᵢⱼ₊₁) * f_ξη * Δt
  uⁿ⁺¹ᵢ₋₁ⱼ₋₁ = (-aᵢ₋₁ⱼ - aᵢⱼ₋₁) * f_ξη * Δt
  uⁿ⁺¹ᵢ₋₁ⱼ₊₁ = (+aᵢ₋₁ⱼ + aᵢⱼ₊₁) * f_ξη * Δt

  stencil = SVector{9,T}(
    uⁿ⁺¹ᵢ₋₁ⱼ₋₁,
    uⁿ⁺¹ᵢ₋₁ⱼ,
    uⁿ⁺¹ᵢ₋₁ⱼ₊₁,
    uⁿ⁺¹ᵢⱼ₋₁,
    uⁿ⁺¹ᵢⱼ,
    uⁿ⁺¹ᵢⱼ₊₁,
    uⁿ⁺¹ᵢ₊₁ⱼ₋₁,
    uⁿ⁺¹ᵢ₊₁ⱼ,
    uⁿ⁺¹ᵢ₊₁ⱼ₊₁,
  )

  return stencil
end

function inner_op_3d(metric_terms, diffusivity, Δt::T) where {T}

  #
  @unpack α, β, γ, f_ξ², f_η², f_ζ², f_ξη, f_ζη, f_ζξ = metric_terms
  @unpack aᵢⱼₖ,
  aᵢ₊₁ⱼₖ, aᵢ₋₁ⱼₖ, aᵢⱼ₊₁ₖ, aᵢⱼ₋₁ₖ, aᵢⱼₖ₊₁, aᵢⱼₖ₋₁, aᵢ₊½, aᵢ₋½, aⱼ₊½, aⱼ₋½, aₖ₊½,
  aₖ₋½ = diffusivity

  uⁿ⁺¹ᵢⱼₖ =
    one(T) + (
      (aᵢ₊½ + aᵢ₋½) * f_ξ² + #
      (aⱼ₊½ + aⱼ₋½) * f_η² + #
      (aₖ₊½ + aₖ₋½) * f_ζ²   #
    ) * Δt

  uⁿ⁺¹ᵢ₊₁ⱼ₊₁ₖ₋₁ = zero(T)
  uⁿ⁺¹ᵢ₊₁ⱼ₊₁ₖ₊₁ = zero(T)
  uⁿ⁺¹ᵢ₊₁ⱼ₋₁ₖ₋₁ = zero(T)
  uⁿ⁺¹ᵢ₊₁ⱼ₋₁ₖ₊₁ = zero(T)
  uⁿ⁺¹ᵢ₋₁ⱼ₊₁ₖ₋₁ = zero(T)
  uⁿ⁺¹ᵢ₋₁ⱼ₊₁ₖ₊₁ = zero(T)
  uⁿ⁺¹ᵢ₋₁ⱼ₋₁ₖ₋₁ = zero(T)
  uⁿ⁺¹ᵢ₋₁ⱼ₋₁ₖ₊₁ = zero(T)

  uⁿ⁺¹ᵢ₊₁ⱼₖ = (-aᵢ₊½ * f_ξ² - aᵢⱼₖ * (α / 2)) * Δt
  uⁿ⁺¹ᵢ₋₁ⱼₖ = (-aᵢ₋½ * f_ξ² + aᵢⱼₖ * (α / 2)) * Δt

  uⁿ⁺¹ᵢⱼ₊₁ₖ = (-aⱼ₊½ * f_η² - aᵢⱼₖ * (β / 2)) * Δt
  uⁿ⁺¹ᵢⱼ₋₁ₖ = (-aⱼ₋½ * f_η² + aᵢⱼₖ * (β / 2)) * Δt

  uⁿ⁺¹ᵢⱼₖ₋₁ = (-aₖ₋½ * f_ζ² + aᵢⱼₖ * (γ / 2)) * Δt
  uⁿ⁺¹ᵢⱼₖ₊₁ = (-aₖ₊½ * f_ζ² - aᵢⱼₖ * (γ / 2)) * Δt

  uⁿ⁺¹ᵢ₋₁ⱼ₊₁ₖ = (aᵢ₋₁ⱼₖ + aᵢⱼ₊₁ₖ) * f_ξη * Δt
  uⁿ⁺¹ᵢ₊₁ⱼ₋₁ₖ = (aᵢ₊₁ⱼₖ + aᵢⱼ₋₁ₖ) * f_ξη * Δt
  uⁿ⁺¹ᵢ₊₁ⱼ₊₁ₖ = (-aᵢ₊₁ⱼₖ - aᵢⱼ₊₁ₖ) * f_ξη * Δt
  uⁿ⁺¹ᵢ₋₁ⱼ₋₁ₖ = (-aᵢ₋₁ⱼₖ - aᵢⱼ₋₁ₖ) * f_ξη * Δt

  uⁿ⁺¹ᵢ₊₁ⱼₖ₋₁ = (aᵢ₊₁ⱼₖ + aᵢⱼₖ₋₁) * f_ζξ * Δt
  uⁿ⁺¹ᵢ₋₁ⱼₖ₊₁ = (aᵢ₋₁ⱼₖ + aᵢⱼₖ₊₁) * f_ζξ * Δt
  uⁿ⁺¹ᵢ₊₁ⱼₖ₊₁ = (-aᵢ₊₁ⱼₖ - aᵢⱼₖ₊₁) * f_ζξ * Δt
  uⁿ⁺¹ᵢ₋₁ⱼₖ₋₁ = (-aᵢ₋₁ⱼₖ - aᵢⱼₖ₋₁) * f_ζξ * Δt

  uⁿ⁺¹ᵢⱼ₊₁ₖ₋₁ = (aᵢⱼ₊₁ₖ + aᵢⱼₖ₋₁) * f_ζη * Δt
  uⁿ⁺¹ᵢⱼ₋₁ₖ₊₁ = (aᵢⱼ₋₁ₖ + aᵢⱼₖ₊₁) * f_ζη * Δt
  uⁿ⁺¹ᵢⱼ₊₁ₖ₊₁ = (-aᵢⱼ₊₁ₖ - aᵢⱼₖ₊₁) * f_ζη * Δt
  uⁿ⁺¹ᵢⱼ₋₁ₖ₋₁ = (-aᵢⱼ₋₁ₖ - aᵢⱼₖ₋₁) * f_ζη * Δt

  stencil = SVector{27,T}(
    uⁿ⁺¹ᵢ₋₁ⱼ₋₁ₖ₋₁,
    uⁿ⁺¹ᵢⱼ₋₁ₖ₋₁,
    uⁿ⁺¹ᵢ₊₁ⱼ₋₁ₖ₋₁,
    uⁿ⁺¹ᵢ₋₁ⱼₖ₋₁,
    uⁿ⁺¹ᵢⱼₖ₋₁,
    uⁿ⁺¹ᵢ₊₁ⱼₖ₋₁,
    uⁿ⁺¹ᵢ₋₁ⱼ₊₁ₖ₋₁,
    uⁿ⁺¹ᵢⱼ₊₁ₖ₋₁,
    uⁿ⁺¹ᵢ₊₁ⱼ₊₁ₖ₋₁,
    uⁿ⁺¹ᵢ₋₁ⱼ₋₁ₖ,
    uⁿ⁺¹ᵢⱼ₋₁ₖ,
    uⁿ⁺¹ᵢ₊₁ⱼ₋₁ₖ,
    uⁿ⁺¹ᵢ₋₁ⱼₖ,
    uⁿ⁺¹ᵢⱼₖ,
    uⁿ⁺¹ᵢ₊₁ⱼₖ,
    uⁿ⁺¹ᵢ₋₁ⱼ₊₁ₖ,
    uⁿ⁺¹ᵢⱼ₊₁ₖ,
    uⁿ⁺¹ᵢ₊₁ⱼ₊₁ₖ,
    uⁿ⁺¹ᵢ₋₁ⱼ₋₁ₖ₊₁,
    uⁿ⁺¹ᵢⱼ₋₁ₖ₊₁,
    uⁿ⁺¹ᵢ₊₁ⱼ₋₁ₖ₊₁,
    uⁿ⁺¹ᵢ₋₁ⱼₖ₊₁,
    uⁿ⁺¹ᵢⱼₖ₊₁,
    uⁿ⁺¹ᵢ₊₁ⱼₖ₊₁,
    uⁿ⁺¹ᵢ₋₁ⱼ₊₁ₖ₊₁,
    uⁿ⁺¹ᵢⱼ₊₁ₖ₊₁,
    uⁿ⁺¹ᵢ₊₁ⱼ₊₁ₖ₊₁,
  )

  return stencil
end
