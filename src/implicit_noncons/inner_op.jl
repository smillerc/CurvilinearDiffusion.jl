using UnPack
using StaticArrays
using OffsetArrays

function inner_op_1d(
  metric_terms, diffusivity, uⁿᵢ::T, Δt::T, s::T, meanfunc::F
) where {T,F<:Function}
  @unpack α, f_ξ² = metric_terms
  @unpack aᵢ, aᵢ₊₁, aᵢ₋₁ = diffusivity

  inv_dt = inv(Δt)

  aᵢ₊½ = meanfunc(aᵢ, aᵢ₊₁)
  aᵢ₋½ = meanfunc(aᵢ, aᵢ₋₁)

  inv_dt = inv(Δt)

  uⁿ⁺¹ᵢ = inv_dt + (aᵢ₊½ + aᵢ₋½) * f_ξ²
  uⁿ⁺¹ᵢ₊₁ = -aᵢ * (α / 2) - aᵢ₊½ * f_ξ²
  uⁿ⁺¹ᵢ₋₁ = +aᵢ * (α / 2) - aᵢ₋½ * f_ξ²

  rhs = s + uⁿᵢ * inv_dt

  stencil = SVector{3,T}(uⁿ⁺¹ᵢ₋₁, uⁿ⁺¹ᵢ, uⁿ⁺¹ᵢ₊₁)

  return OffsetVector(stencil, -1:1), rhs
end

function inner_op_2d(
  metric_terms, diffusivity, uⁿᵢⱼ::T, Δt::T, s::T, meanfunc::F
) where {T,F<:Function}
  @unpack α, β, f_ξ², f_η², f_ξη = metric_terms
  @unpack aᵢⱼ, aᵢ₊₁ⱼ, aᵢ₋₁ⱼ, aᵢⱼ₊₁, aᵢⱼ₋₁ = diffusivity

  aᵢ₊½ = meanfunc(aᵢⱼ, aᵢ₊₁ⱼ)
  aᵢ₋½ = meanfunc(aᵢⱼ, aᵢ₋₁ⱼ)
  aⱼ₊½ = meanfunc(aᵢⱼ, aᵢⱼ₊₁)
  aⱼ₋½ = meanfunc(aᵢⱼ, aᵢⱼ₋₁)

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

function inner_op_3d(
  metric_terms, diffusivity, uⁿᵢⱼₖ::T, Δt::T, s::T, meanfunc::F
) where {T,F<:Function}
  @unpack α, β, f_ξ², f_η², f_ζ², f_ξη, f_ζη, f_ζξ = metric_terms
  @unpack aᵢⱼₖ, aᵢ₊₁ⱼₖ, aᵢ₋₁ⱼₖ, aᵢⱼ₊₁ₖ, aᵢⱼ₋₁ₖ, aᵢⱼₖ₊₁, aᵢⱼₖ₋₁ = diffusivity

  inv_dt = inv(Δt)

  aᵢ₊½ = meanfunc(aᵢⱼₖ, aᵢ₊₁ⱼₖ)
  aᵢ₋½ = meanfunc(aᵢⱼₖ, aᵢ₋₁ⱼₖ)
  aⱼ₊½ = meanfunc(aᵢⱼₖ, aᵢⱼ₊₁ₖ)
  aⱼ₋½ = meanfunc(aᵢⱼₖ, aᵢⱼ₋₁ₖ)
  aₖ₊½ = meanfunc(aᵢⱼₖ, aᵢⱼₖ₊₁)
  aₖ₋½ = meanfunc(aᵢⱼₖ, aᵢⱼₖ₋₁)

  rhs = s + uⁿᵢⱼₖ * inv_dt

  stencil = SArray{Tuple{3,3,3},T}(
    0,                             # uⁿ⁺¹ᵢ₋₁ⱼ₋₁ₖ₋₁ 
    (-aᵢⱼ₋₁ₖ - aᵢⱼₖ₋₁) * f_ζη,     # uⁿ⁺¹ᵢⱼ₋₁ₖ₋₁ 
    0,                             # uⁿ⁺¹ᵢ₊₁ⱼ₋₁ₖ₋₁ 
    (-aᵢ₋₁ⱼₖ - aᵢⱼₖ₋₁) * f_ζξ,     # uⁿ⁺¹ᵢ₋₁ⱼₖ₋₁
    (+aᵢⱼₖ * γ / 2 - aₖ₋½ * f_ζ²), # uⁿ⁺¹ᵢⱼₖ₋₁
    (aᵢ₊₁ⱼₖ + aᵢⱼₖ₋₁) * f_ζξ,      # uⁿ⁺¹ᵢ₊₁ⱼₖ₋₁
    0,                             # uⁿ⁺¹ᵢ₋₁ⱼ₊₁ₖ₋₁ 
    (aᵢⱼ₊₁ₖ + aᵢⱼₖ₋₁) * f_ζη,      # uⁿ⁺¹ᵢⱼ₊₁ₖ₋₁ 
    0,                             # uⁿ⁺¹ᵢ₊₁ⱼ₊₁ₖ₋₁ 
    (-aᵢ₋₁ⱼₖ - aᵢⱼ₋₁ₖ) * f_ξη,     # uⁿ⁺¹ᵢ₋₁ⱼ₋₁ₖ
    (+aᵢⱼₖ * β / 2 - aⱼ₋½ * f_η²), # uⁿ⁺¹ᵢⱼ₋₁ₖ
    (aᵢ₊₁ⱼₖ + aᵢⱼ₋₁ₖ) * f_ξη,      # uⁿ⁺¹ᵢ₊₁ⱼ₋₁ₖ
    (+aᵢⱼₖ * α / 2 - aᵢ₋½ * f_ξ²), # uⁿ⁺¹ᵢ₋₁ⱼₖ
    inv(Δt) + ((aᵢ₊½ + aᵢ₋½) * f_ξ² + (aₖ₊½ + aₖ₋½) * f_ζ² + (aⱼ₊½ + aⱼ₋½) * f_η²), # uⁿ⁺¹ᵢⱼₖ
    (-aᵢⱼₖ * α / 2 - aᵢ₊½ * f_ξ²), # uⁿ⁺¹ᵢ₊₁ⱼₖ
    (aᵢ₋₁ⱼₖ + aᵢⱼ₊₁ₖ) * f_ξη,      # uⁿ⁺¹ᵢ₋₁ⱼ₊₁ₖ
    (-aᵢⱼₖ * β / 2 - aⱼ₊½ * f_η²), # uⁿ⁺¹ᵢⱼ₊₁ₖ
    (-aᵢ₊₁ⱼₖ - aᵢⱼ₊₁ₖ) * f_ξη,     # uⁿ⁺¹ᵢ₊₁ⱼ₊₁ₖ
    0,                             # uⁿ⁺¹ᵢ₋ⱼ₋₁ₖ₊₁ 
    (aᵢⱼ₋₁ₖ + aᵢⱼₖ₊₁) * f_ζη,      # uⁿ⁺¹ᵢⱼ₋₁ₖ₊₁ 
    0,                             # uⁿ⁺¹ᵢ₊ⱼ₋₁ₖ₊₁ 
    (aᵢ₋₁ⱼₖ + aᵢⱼₖ₊₁) * f_ζξ,      # uⁿ⁺¹ᵢ₋₁ⱼₖ₊₁
    (-aᵢⱼₖ * γ / 2 - aₖ₊½ * f_ζ²), # uⁿ⁺¹ᵢⱼₖ₊₁
    (-aᵢ₊₁ⱼₖ - aᵢⱼₖ₊₁) * f_ζξ,     # uⁿ⁺¹ᵢ₊₁ⱼₖ₊₁
    0,                             # uⁿ⁺¹ᵢ₋₁ⱼ₊₁ₖ₊₁ 
    (-aᵢⱼ₊₁ₖ - aᵢⱼₖ₊₁) * f_ζη,     # uⁿ⁺¹ᵢⱼ₊₁ₖ₊₁ 
    0,                             # uⁿ⁺¹ᵢ₊₁ⱼ₊₁ₖ₊₁ 
  )
  return OffsetArray(stencil, (-1:1, -1:1, -1:1)), rhs
end
