using UnPack

# @testset "1D"
begin
  include("../../src/implicit/inner_op.jl")
  α, f_ξ² = [0, 1]
  aᵢ, aᵢ₊₁, aᵢ₋₁ = ones(3)

  metric_terms = (; α, f_ξ²)
  diffusivity = (; aᵢ, aᵢ₊₁, aᵢ₋₁)
  u = 1.0
  s = 0.0
  Δt = 1e-0
  meanfunc(a, b) = 0.5(a + b)
  inner_op_1d(metric_terms, diffusivity, u, Δt, s, meanfunc)
end

begin
  include("../../src/implicit/inner_op.jl")

  α, β, f_ξ², f_η², f_ξη = [0, 0, 1, 1, 0]
  aᵢⱼ, aᵢ₊₁ⱼ, aᵢ₋₁ⱼ, aᵢⱼ₊₁, aᵢⱼ₋₁ = ones(5)

  metric_terms = (; α, β, f_ξ², f_η², f_ξη)
  diffusivity = (; aᵢⱼ, aᵢ₊₁ⱼ, aᵢ₋₁ⱼ, aᵢⱼ₊₁, aᵢⱼ₋₁)
  u = 1.0
  s = 0.0
  Δt = Inf
  meanfunc(a, b) = 0.5(a + b)
  stencil, rhs = inner_op_2d(metric_terms, diffusivity, u, Δt, s, meanfunc)
  display(stencil)
end

begin
  include("../../src/implicit/inner_op.jl")

  α, β, γ, f_ξ², f_η², f_ζ², f_ξη, f_ζη, f_ζξ = [0, 0, 0, 1, 1, 1, 0, 0, 0]
  aᵢⱼₖ, aᵢ₊₁ⱼₖ, aᵢ₋₁ⱼₖ, aᵢⱼ₊₁ₖ, aᵢⱼ₋₁ₖ, aᵢⱼₖ₊₁, aᵢⱼₖ₋₁ = ones(7)

  metric_terms = (; α, β, f_ξ², f_η², f_ζ², f_ξη, f_ζη, f_ζξ)
  diffusivity = (; aᵢⱼₖ, aᵢ₊₁ⱼₖ, aᵢ₋₁ⱼₖ, aᵢⱼ₊₁ₖ, aᵢⱼ₋₁ₖ, aᵢⱼₖ₊₁, aᵢⱼₖ₋₁)
  u = 1.0
  s = 0.0
  Δt = Inf
  meanfunc(a, b) = 0.5(a + b)
  stencil, rhs = inner_op_3d(metric_terms, diffusivity, u, Δt, s, meanfunc)
  display(stencil)
end