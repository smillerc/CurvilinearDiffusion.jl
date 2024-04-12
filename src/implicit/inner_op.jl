function inner_op_1d()
  uᵢ = (-aᵢ₊½ⱼ * f_ξ - aᵢ₋½ⱼ * f_ξ) + sᵢ
  uᵢ₊₁ = (+aᵢ₊½ⱼ * f_ξ + 0.5α)
  uᵢ₋₁ = (+aᵢ₋½ⱼ * f_ξ - 0.5α)

  stencil = @SVector [
    uᵢ₋₁ uᵢ uᵢ₊₁
  ]

  return stencil
end

function inner_op_2d(metric_terms)
  @unpack α, β, f_ξ, f_η, f_ξη = metric_terms

  uᵢⱼ = (
    -aᵢ₊½ⱼ * f_ξ - aᵢ₋½ⱼ * f_ξ  #
    - aᵢⱼ₊½ * f_η - aᵢⱼ₋½ * f_η #
  ) + sᵢⱼ

  uᵢ₊₁ⱼ = (+aᵢ₊½ⱼ * f_ξ + 0.5α)
  uᵢ₋₁ⱼ = (+aᵢ₋½ⱼ * f_ξ - 0.5α)
  uᵢⱼ₊₁ = (+aᵢⱼ₊½ * f_η + 0.5β)
  uᵢⱼ₋₁ = (+aᵢⱼ₋½ * f_η - 0.5β)
  uᵢ₊₁ⱼ₊₁ = (+aᵢ₊₁ⱼ * f_ξη + aᵢⱼ₊₁ * f_ξη)
  uᵢ₊₁ⱼ₋₁ = (-aᵢ₊₁ⱼ * f_ξη - aᵢⱼ₋₁ * f_ξη)
  uᵢ₋₁ⱼ₊₁ = (-aᵢ₋₁ⱼ * f_ξη - aᵢⱼ₊₁ * f_ξη)
  uᵢ₋₁ⱼ₋₁ = (+aᵢ₋₁ⱼ * f_ξη + aᵢⱼ₋₁ * f_ξη)

  #! format: off
  stencil = @SMatrix [
    uᵢ₋₁ⱼ₋₁ uᵢⱼ₋₁ uᵢ₊₁ⱼ₋₁
    uᵢ₋₁ⱼ   uᵢⱼ   uᵢ₊₁ⱼ
    uᵢ₋₁ⱼ₊₁ uᵢⱼ₊₁ uᵢ₊₁ⱼ₊₁
  ]
  #! format: on

  return stencil
end

function inner_op_3d(metric_terms)
  @unpack α, β, f_ξ, f_η, f_ξη = metric_terms

  uᵢⱼₖ =
    (
      -aᵢ₊½ⱼₖ * f_ξ - aᵢ₋½ⱼₖ * f_ξ #
      - aᵢⱼ₊½ₖ * f_η - aᵢⱼ₋½ₖ * f_η #
      - aᵢⱼₖ₊½ * f_ζ - aᵢⱼₖ₋½ * f_ζ #
    ) + sᵢⱼₖ

  uᵢ₊₁ⱼₖ = (+aᵢ₊½ⱼ * f_ξ + 0.5α)
  uᵢ₋₁ⱼₖ = (+aᵢ₋½ⱼ * f_ξ - 0.5α)
  uᵢⱼ₊₁ₖ = (+aᵢⱼ₊½ * f_η + 0.5β)
  uᵢⱼ₋₁ₖ = (+aᵢⱼ₋½ * f_η - 0.5β)
  uᵢⱼₖ₊₁ = (+aᵢⱼₖ₊½ * f_ζ + 0.5γ)
  uᵢⱼₖ₋₁ = (+aᵢⱼₖ₋½ * f_ζ - 0.5γ)

  uᵢ₊₁ⱼ₊₁ = (+aᵢ₊₁ⱼ * f_2 + aᵢⱼ₊₁ * f_3)
  uᵢ₊₁ⱼ₋₁ = (-aᵢ₊₁ⱼ * f_2 - aᵢⱼ₋₁ * f_3)
  uᵢ₋₁ⱼ₊₁ = (-aᵢ₋₁ⱼ * f_2 - aᵢⱼ₊₁ * f_3)
  uᵢ₋₁ⱼ₋₁ = (+aᵢ₋₁ⱼ * f_2 + aᵢⱼ₋₁ * f_3)

  return nothing
end