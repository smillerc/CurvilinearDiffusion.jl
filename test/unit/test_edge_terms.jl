include("common.jl")

@testset "1d edge terms" begin
  include("../../src/edge_terms.jl")

  edge_diffusivity = (αᵢ₊½=1.0, αᵢ₋½=2.0)
  edge_metrics = (Jᵢ₊½=1.0, Jᵢ₋½=2.0, Jξx_ᵢ₊½=3.0, Jξx_ᵢ₋½=4.0)

  edge_terms = conservative_edge_terms(edge_diffusivity, edge_metrics)

  # Ensure that it doesn't allocate! This would kill performance and indicate
  # a type instability somewhere
  bm1 = @benchmark conservative_edge_terms($edge_diffusivity, $edge_metrics)
  @test bm1.allocs == 0

  edge_terms = conservative_edge_terms(edge_diffusivity, edge_metrics)
  @test edge_terms.fᵢ₊½ == 9.0
  @test edge_terms.fᵢ₋½ == 16.0
end

@testset "2d edge terms" begin
  include("../../src/edge_terms.jl")

  edge_diffusivity = (αᵢ₊½=1.0, αᵢ₋½=2.0, αⱼ₊½=3.0, αⱼ₋½=4.0)
  non_orth_edge_metrics = (
    Jᵢ₊½=1.0,
    Jᵢ₋½=2.0,
    Jⱼ₊½=3.0,
    Jⱼ₋½=4.0,
    Jξx_ᵢ₊½=5.0,
    Jξy_ᵢ₊½=6.0,
    Jηx_ᵢ₊½=7.0,
    Jηy_ᵢ₊½=8.0,
    Jξx_ᵢ₋½=9.0,
    Jξy_ᵢ₋½=10.0,
    Jηx_ᵢ₋½=11.0,
    Jηy_ᵢ₋½=12.0,
    Jξx_ⱼ₊½=13.0,
    Jξy_ⱼ₊½=14.0,
    Jηx_ⱼ₊½=15.0,
    Jηy_ⱼ₊½=16.0,
    Jξx_ⱼ₋½=17.0,
    Jξy_ⱼ₋½=18.0,
    Jηx_ⱼ₋½=19.0,
    Jηy_ⱼ₋½=20.0,
  )
  orth_edge_metrics = (
    Jᵢ₊½=1.0,
    Jᵢ₋½=2.0,
    Jⱼ₊½=3.0,
    Jⱼ₋½=4.0,
    Jξx_ᵢ₊½=5.0,
    Jηy_ᵢ₊½=8.0,
    Jξx_ᵢ₋½=9.0,
    Jηy_ᵢ₋½=12.0,
    Jξx_ⱼ₊½=13.0,
    Jηy_ⱼ₊½=16.0,
    Jξx_ⱼ₋½=17.0,
    Jηy_ⱼ₋½=20.0,
    Jξy_ᵢ₊½=0.0,
    Jξy_ᵢ₋½=0.0,
    Jξy_ⱼ₊½=0.0,
    Jξy_ⱼ₋½=0.0,
    Jηx_ᵢ₊½=0.0,
    Jηx_ᵢ₋½=0.0,
    Jηx_ⱼ₊½=0.0,
    Jηx_ⱼ₋½=0.0,
  )
  non_orth_edge_terms = conservative_edge_terms(edge_diffusivity, non_orth_edge_metrics)

  # Ensure that it doesn't allocate! This would kill performance and indicate
  # a type instability somewhere
  bm1 = @benchmark conservative_edge_terms($edge_diffusivity, $non_orth_edge_metrics)
  @test bm1.allocs == 0

  orth_edge_terms = conservative_edge_terms(edge_diffusivity, orth_edge_metrics)

  @test non_orth_edge_terms.fᵢ₊½ == 61.0
  @test non_orth_edge_terms.fᵢ₋½ == 181.0
  @test non_orth_edge_terms.fⱼ₊½ == 481.0
  @test non_orth_edge_terms.fⱼ₋½ == 761.0
  @test non_orth_edge_terms.gᵢ₊½ == 20.75
  @test non_orth_edge_terms.gᵢ₋½ == 54.75
  @test non_orth_edge_terms.gⱼ₊½ == 104.75
  @test non_orth_edge_terms.gⱼ₋½ == 170.75

  @test orth_edge_terms.fᵢ₊½ == 25.0
  @test orth_edge_terms.fᵢ₋½ == 81.0
  @test orth_edge_terms.fⱼ₊½ == 256.0
  @test orth_edge_terms.fⱼ₋½ == 400.0
  @test orth_edge_terms.gᵢ₊½ == 0.0
  @test orth_edge_terms.gᵢ₋½ == 0.0
  @test orth_edge_terms.gⱼ₊½ == 0.0
  @test orth_edge_terms.gⱼ₋½ == 0.0
end

@testset "3d edge terms" begin
  include("../../src/edge_terms.jl")

  edge_diffusivity = (αᵢ₊½=1.0, αᵢ₋½=2.0, αⱼ₊½=3.0, αⱼ₋½=4.0, αₖ₊½=5.0, αₖ₋½=6.0)
  non_orth_edge_metrics = (
    Jᵢ₊½=1.0,
    Jᵢ₋½=2.0,
    Jⱼ₊½=3.0,
    Jⱼ₋½=4.0,
    Jₖ₊½=3.0,
    Jₖ₋½=4.0,
    Jξx_ᵢ₊½=5.0,
    Jξy_ᵢ₊½=6.0,
    Jξz_ᵢ₊½=6.0,
    Jηx_ᵢ₊½=7.0,
    Jηy_ᵢ₊½=8.0,
    Jηz_ᵢ₊½=8.0,
    Jζx_ᵢ₊½=7.0,
    Jζy_ᵢ₊½=8.0,
    Jζz_ᵢ₊½=8.0,
    Jξx_ᵢ₋½=9.0,
    Jξy_ᵢ₋½=10.0,
    Jξz_ᵢ₋½=10.0,
    Jηx_ᵢ₋½=11.0,
    Jηy_ᵢ₋½=12.0,
    Jηz_ᵢ₋½=12.0,
    Jζx_ᵢ₋½=11.0,
    Jζy_ᵢ₋½=12.0,
    Jζz_ᵢ₋½=12.0,
    Jξx_ⱼ₊½=13.0,
    Jξy_ⱼ₊½=14.0,
    Jξz_ⱼ₊½=14.0,
    Jηx_ⱼ₊½=15.0,
    Jηy_ⱼ₊½=16.0,
    Jηz_ⱼ₊½=16.0,
    Jζx_ⱼ₊½=15.0,
    Jζy_ⱼ₊½=16.0,
    Jζz_ⱼ₊½=16.0,
    Jξx_ⱼ₋½=17.0,
    Jξy_ⱼ₋½=18.0,
    Jξz_ⱼ₋½=18.0,
    Jηx_ⱼ₋½=19.0,
    Jηy_ⱼ₋½=20.0,
    Jηz_ⱼ₋½=20.0,
    Jζx_ⱼ₋½=19.0,
    Jζy_ⱼ₋½=20.0,
    Jζz_ⱼ₋½=20.0,
    Jξx_ₖ₊½=13.0,
    Jξy_ₖ₊½=14.0,
    Jξz_ₖ₊½=14.0,
    Jηx_ₖ₊½=15.0,
    Jηy_ₖ₊½=16.0,
    Jηz_ₖ₊½=16.0,
    Jζx_ₖ₊½=15.0,
    Jζy_ₖ₊½=16.0,
    Jζz_ₖ₊½=16.0,
    Jξx_ₖ₋½=17.0,
    Jξy_ₖ₋½=18.0,
    Jξz_ₖ₋½=18.0,
    Jηx_ₖ₋½=19.0,
    Jηy_ₖ₋½=20.0,
    Jηz_ₖ₋½=20.0,
    Jζx_ₖ₋½=19.0,
    Jζy_ₖ₋½=20.0,
    Jζz_ₖ₋½=20.0,
  )

  non_orth_edge_terms = conservative_edge_terms(edge_diffusivity, non_orth_edge_metrics)

  # Ensure that it doesn't allocate! This would kill performance and indicate
  # a type instability somewhere
  bm1 = @benchmark conservative_edge_terms($edge_diffusivity, $non_orth_edge_metrics)
  @test bm1.allocs == 0

  # orth_edge_terms = conservative_edge_terms(edge_diffusivity, orth_edge_metrics)

end
