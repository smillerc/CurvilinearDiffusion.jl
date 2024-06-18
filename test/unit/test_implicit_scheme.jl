
@testset "ImplicitScheme 2D construction" begin
  x0, x1 = (-6, 6)
  y0, y1 = (-6, 6)

  ni, nj = (6, 6)
  nhalo = 2

  mesh = RectlinearGrid((x0, y0), (x1, y1), (ni, nj), nhalo)

  bcs = (ilo=NeumannBC(), ihi=NeumannBC(), jlo=NeumannBC(), jhi=NeumannBC())
  scheme = ImplicitScheme(mesh, bcs)

  @test scheme.iterators.mesh == CartesianIndices((2:9, 2:9))
  @test scheme.iterators.full.cartesian == CartesianIndices((8, 8))
  @test scheme.iterators.domain.cartesian == CartesianIndices((2:7, 2:7))
  @test mesh.iterators.cell.domain == CartesianIndices((3:8, 3:8))
  @test mesh.iterators.cell.full == CartesianIndices((10, 10))

  @test length(scheme.linear_problem.b) == 64
  @test size(scheme.linear_problem.A) == (64, 64)
  @test size(scheme.α) == (8, 8)
  @test size(scheme.source_term) == (8, 8)
  @test scheme.limits == (ilo=1, jlo=1, ihi=8, jhi=8)
  @test size(mesh.iterators.cell.domain) == size(scheme.iterators.domain.linear)
end

@testset "ImplicitScheme 3D construction" begin
  x0, x1 = (-6, 6)
  y0, y1 = (-6, 6)
  z0, z1 = (-6, 6)
  ni = nj = nk = 6
  nhalo = 4

  mesh = RectlinearGrid((x0, y0, z0), (x1, y1, z1), (ni, nj, nk), nhalo)

  bcs = (
    ilo=NeumannBC(),
    ihi=NeumannBC(),
    jlo=NeumannBC(),
    jhi=NeumannBC(),
    klo=NeumannBC(),
    khi=NeumannBC(),
  )
  scheme = ImplicitScheme(mesh, bcs)

  @test mesh.iterators.cell.domain == CartesianIndices((5:10, 5:10, 5:10))

  @test scheme.iterators.mesh == CartesianIndices((4:11, 4:11, 4:11))
  @test scheme.iterators.full.cartesian == CartesianIndices((8, 8, 8))
  @test scheme.iterators.domain.cartesian == CartesianIndices((2:7, 2:7, 2:7))
  @test mesh.iterators.cell.full == CartesianIndices((14, 14, 14))

  len = 8 * 8 * 8
  @test length(scheme.linear_problem.b) == len
  @test size(scheme.linear_problem.A) == (len, len)
  @test size(scheme.α) == (8, 8, 8)
  @test size(scheme.source_term) == (8, 8, 8)
  @test scheme.limits == (ilo=1, jlo=1, klo=1, ihi=8, jhi=8, khi=8)
  @test size(mesh.iterators.cell.domain) == size(scheme.iterators.domain.linear)
end
