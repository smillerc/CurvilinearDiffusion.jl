using UnPack

@testset "1D inner operator" begin
  using CurvilinearDiffusion.ImplicitSchemeType: inner_op_1d

  α, f_ξ² = [0, 1]
  aᵢ, aᵢ₊₁, aᵢ₋₁ = ones(3)
  meanfunc(a, b) = 0.5(a + b)

  aᵢ₊½ = meanfunc(aᵢ, aᵢ₊₁)
  aᵢ₋½ = meanfunc(aᵢ, aᵢ₋₁)

  metric_terms = (; α, f_ξ²)
  diffusivity = (; aᵢ, aᵢ₊₁, aᵢ₋₁, aᵢ₋½, aᵢ₊½)
  Δt = 1.0
  stencil = inner_op_1d(metric_terms, diffusivity, Δt)

  s = SVector{3,Float64}(-1, 3, -1)

  @test stencil == s
  bm = @benchmark inner_op_1d($metric_terms, $diffusivity, $Δt)
  @test bm.allocs == 0
end

@testset "2D inner operator" begin
  using CurvilinearDiffusion.ImplicitSchemeType: inner_op_2d

  α, β, f_ξ², f_η², f_ξη = [0, 0, 1, 1, 0]
  aᵢⱼ, aᵢ₊₁ⱼ, aᵢ₋₁ⱼ, aᵢⱼ₊₁, aᵢⱼ₋₁ = ones(5)
  meanfunc(a, b) = 0.5(a + b)

  aᵢ₊½ = meanfunc(aᵢⱼ, aᵢ₊₁ⱼ)
  aᵢ₋½ = meanfunc(aᵢⱼ, aᵢ₋₁ⱼ)
  aⱼ₊½ = meanfunc(aᵢⱼ, aᵢⱼ₊₁)
  aⱼ₋½ = meanfunc(aᵢⱼ, aᵢⱼ₋₁)

  metric_terms = (; α, β, f_ξ², f_η², f_ξη)
  diffusivity = (; aᵢⱼ, aᵢ₊₁ⱼ, aᵢ₋₁ⱼ, aᵢⱼ₊₁, aᵢⱼ₋₁, aᵢ₊½, aᵢ₋½, aⱼ₊½, aⱼ₋½)
  u = 1.0
  s = 0.0
  Δt = 1.0
  stencil = inner_op_2d(metric_terms, diffusivity, Δt)

  s = SVector{9,Float64}(0, -1, 0, -1, 5, -1, 0, -1, 0)

  @test vec(stencil) == s

  bm = @benchmark inner_op_2d($metric_terms, $diffusivity, $Δt)
  @test bm.allocs == 0
end

@testset "3D inner operator" begin
  using CurvilinearDiffusion.ImplicitSchemeType: inner_op_3d

  α, β, γ, f_ξ², f_η², f_ζ², f_ξη, f_ζη, f_ζξ = [0, 0, 0, 1, 1, 1, 0, 0, 0]
  aᵢⱼₖ, aᵢ₊₁ⱼₖ, aᵢ₋₁ⱼₖ, aᵢⱼ₊₁ₖ, aᵢⱼ₋₁ₖ, aᵢⱼₖ₊₁, aᵢⱼₖ₋₁ = ones(7)
  meanfunc(a, b) = 0.5(a + b)

  aᵢ₊½ = meanfunc(aᵢⱼₖ, aᵢ₊₁ⱼₖ)
  aᵢ₋½ = meanfunc(aᵢⱼₖ, aᵢ₋₁ⱼₖ)
  aⱼ₊½ = meanfunc(aᵢⱼₖ, aᵢⱼ₊₁ₖ)
  aⱼ₋½ = meanfunc(aᵢⱼₖ, aᵢⱼ₋₁ₖ)
  aₖ₊½ = meanfunc(aᵢⱼₖ, aᵢⱼ₊₁ₖ)
  aₖ₋½ = meanfunc(aᵢⱼₖ, aᵢⱼ₋₁ₖ)

  metric_terms = (; α, β, γ, f_ξ², f_η², f_ζ², f_ξη, f_ζη, f_ζξ)
  diffusivity = (;
    aᵢⱼₖ, aᵢ₊₁ⱼₖ, aᵢ₋₁ⱼₖ, aᵢⱼ₊₁ₖ, aᵢⱼ₋₁ₖ, aᵢⱼₖ₊₁, aᵢⱼₖ₋₁, aᵢ₊½, aᵢ₋½, aⱼ₊½, aⱼ₋½, aₖ₊½, aₖ₋½
  )
  u = 1.0
  s = 0.0
  Δt = 1
  meanfunc(a, b) = 0.5(a + b)
  stencil = inner_op_3d(metric_terms, diffusivity, Δt)

  s = SVector{27,Float64}(
    0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1, 7, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0
  )

  @test vec(stencil) == s
  bm = @benchmark inner_op_3d($metric_terms, $diffusivity, $Δt)
  @test bm.allocs == 0
end