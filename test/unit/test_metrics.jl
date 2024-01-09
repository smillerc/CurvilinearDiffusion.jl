include("common.jl")

# @testset "1d metric terms" begin
#   include("../../src/mesh_metrics.jl")

#   function rect_grid(nx)
#     x0, x1 = (0, 2)

#     x(ξ) = @. x0 + (x1 - x0) * ((ξ - 1) / (nx - 1))
#     return x
#   end

#   ni = 5
#   nhalo = 0
#   x = rect_grid(ni)
#   mesh = CurvilinearGrid1D(x, ni, nhalo)

#   idx = (3,)
#   m = _conservative_metrics(mesh, idx)

#   # Ensure that it doesn't allocate! This would kill performance and indicate
#   # a type instability somewhere
#   bm1 = @benchmark _conservative_metrics($mesh, $idx)
#   @test bm1.allocs == 0

#   @test m.Jᵢ₊½ == m.Jᵢ₋½ == 0.125
#   @test m.Jξx_ᵢ₋½ == m.Jξx_ᵢ₊½ == 0.25
# end

@testset "2d metric terms" begin
  include("../../src/mesh_metrics.jl")

  function rect_grid(nx, ny)
    x0, x1 = (0, 2)
    y0, y1 = (1, 3)

    x(ξ, η) = @. x0 + (x1 - x0) * ((ξ - 1) / (nx - 1))
    y(ξ, η) = @. y0 + (y1 - y0) * ((η - 1) / (ny - 1))
    return (x, y)
  end

  ni, nj = (5, 9)
  nhalo = 0
  x, y = rect_grid(ni, nj)
  mesh = CurvilinearGrid2D(x, y, (ni, nj), nhalo)

  idx = (3, 4)
  m = _conservative_metrics(mesh, idx)

  # Ensure that it doesn't allocate! This would kill performance and indicate
  # a type instability somewhere
  bm1 = @benchmark _conservative_metrics($mesh, $idx)
  @test bm1.allocs == 0

  @test m.Jᵢ₊½ == m.Jⱼ₊½ == m.Jᵢ₋½ == m.Jⱼ₋½ == 0.125
  @test m.Jξx_ᵢ₋½ == m.Jξx_ⱼ₋½ == m.Jξx_ᵢ₊½ == m.Jξx_ⱼ₊½ == 0.25
  @test m.Jξy_ᵢ₋½ == m.Jξy_ⱼ₋½ == m.Jξy_ᵢ₊½ == m.Jξy_ⱼ₊½ == 0.0 # this is an orthogonal grid
  @test m.Jηx_ᵢ₋½ == m.Jηx_ⱼ₋½ == m.Jηx_ᵢ₊½ == m.Jηx_ⱼ₊½ == 0.0 # this is an orthogonal grid
  @test m.Jηy_ᵢ₋½ == m.Jηy_ⱼ₋½ == m.Jηy_ᵢ₊½ == m.Jηy_ⱼ₊½ == 0.5
end

@testset "3d metric terms" begin
  include("../../src/mesh_metrics.jl")

  function rect_grid(nx, ny, nz)
    x0, x1 = (0, 2)
    y0, y1 = (1, 3)
    z0, z1 = (-2, 2)

    x(ξ, η, ζ) = @. x0 + (x1 - x0) * ((ξ - 1) / (nx - 1))
    y(ξ, η, ζ) = @. y0 + (y1 - y0) * ((η - 1) / (ny - 1))
    z(ξ, η, ζ) = @. z0 + (z1 - z0) * ((ζ - 1) / (nz - 1))
    return (x, y, z)
  end

  ni, nj, nk = (5, 9, 21)
  nhalo = 0
  x, y, z = rect_grid(ni, nj, nk)
  mesh = CurvilinearGrid3D(x, y, z, (ni, nj, nk), nhalo)

  idx = (3, 4, 7)
  m = _conservative_metrics(mesh, idx)

  # Ensure that it doesn't allocate! This would kill performance and indicate
  # a type instability somewhere
  bm1 = @benchmark _conservative_metrics($mesh, $idx)
  @test bm1.allocs == 0

  @test m.Jᵢ₊½ == m.Jⱼ₊½ == m.Jₖ₊½ == m.Jᵢ₋½ == m.Jⱼ₋½ == m.Jₖ₋½ == 0.025

  @test m.Jξx_ᵢ₋½ == m.Jξx_ⱼ₋½ == m.Jξx_ₖ₋½ == m.Jξx_ᵢ₊½ == m.Jξx_ⱼ₊½ == m.Jξx_ₖ₊½ == 0.05
  @test m.Jξy_ᵢ₋½ == m.Jξy_ⱼ₋½ == m.Jξy_ₖ₋½ == m.Jξy_ᵢ₊½ == m.Jξy_ⱼ₊½ == m.Jξy_ₖ₊½ == 0.0
  @test m.Jξz_ᵢ₋½ == m.Jξz_ⱼ₋½ == m.Jξz_ₖ₋½ == m.Jξz_ᵢ₊½ == m.Jξz_ⱼ₊½ == m.Jξz_ₖ₊½ == 0.0
  @test m.Jηx_ᵢ₋½ == m.Jηx_ⱼ₋½ == m.Jηx_ₖ₋½ == m.Jηx_ᵢ₊½ == m.Jηx_ⱼ₊½ == m.Jηx_ₖ₊½ == 0.0
  @test m.Jηy_ᵢ₋½ == m.Jηy_ⱼ₋½ == m.Jηy_ₖ₋½ == m.Jηy_ᵢ₊½ == m.Jηy_ⱼ₊½ == m.Jηy_ₖ₊½ == 0.1
  @test m.Jηz_ᵢ₋½ == m.Jηz_ⱼ₋½ == m.Jηz_ₖ₋½ == m.Jηz_ᵢ₊½ == m.Jηz_ⱼ₊½ == m.Jηz_ₖ₊½ == 0.0
  @test m.Jζx_ᵢ₋½ == m.Jζx_ⱼ₋½ == m.Jζx_ₖ₋½ == m.Jζx_ᵢ₊½ == m.Jζx_ⱼ₊½ == m.Jζx_ₖ₊½ == 0.0
  @test m.Jζy_ᵢ₋½ == m.Jζy_ⱼ₋½ == m.Jζy_ₖ₋½ == m.Jζy_ᵢ₊½ == m.Jζy_ⱼ₊½ == m.Jζy_ₖ₊½ == 0.0
  @test m.Jζz_ᵢ₋½ == m.Jζz_ⱼ₋½ == m.Jζz_ₖ₋½ == m.Jζz_ᵢ₊½ == m.Jζz_ⱼ₊½ == m.Jζz_ₖ₊½ == 0.125
end