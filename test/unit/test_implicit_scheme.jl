
@testset "ImplicitScheme 2D construction" begin
  function uniform_grid(nx, ny)
    x0, x1 = (-6, 6)
    y0, y1 = (-6, 6)

    x(i, j) = @. x0 + (x1 - x0) * ((i - 1) / (nx - 1))
    y(i, j) = @. y0 + (y1 - y0) * ((j - 1) / (ny - 1))

    return (x, y)
  end

  function initialize_mesh()
    ni, nj = (7, 7)
    nhalo = 2
    x, y = uniform_grid(ni, nj)
    return CurvilinearGrid2D(x, y, (ni, nj), nhalo)
  end

  mesh = initialize_mesh()

  bcs = (ilo=NeumannBC(), ihi=NeumannBC(), jlo=NeumannBC(), jhi=NeumannBC())
  scheme = ImplicitScheme(mesh, bcs)

  @test scheme.iterators.mesh == CartesianIndices((2:9, 2:9))
  @test scheme.iterators.full.cartesian == CartesianIndices((8, 8))
  @test scheme.iterators.domain.cartesian == CartesianIndices((2:7, 2:7))
  @test mesh.iterators.cell.domain == CartesianIndices((3:8, 3:8))
  @test mesh.iterators.cell.full == CartesianIndices((10, 10))

  @test length(scheme.b) == 64
  @test length(scheme.x) == 64
  @test length(scheme.solver.x) == 64
  @test size(scheme.A) == (64, 64)
  @test size(scheme.α) == (8, 8)
  @test size(scheme.source_term) == (8, 8)
  @test scheme.limits == (ilo=1, jlo=1, ihi=8, jhi=8)
  @test size(mesh.iterators.cell.domain) == size(scheme.iterators.domain.linear)
end

@testset "ImplicitScheme 3D construction" begin
  function uniform_grid(nx, ny, nz)
    x0, x1 = (-6, 6)
    y0, y1 = (-6, 6)
    z0, z1 = (-6, 6)

    x(i, j, k) = @. x0 + (x1 - x0) * ((i - 1) / (nx - 1))
    y(i, j, k) = @. y0 + (y1 - y0) * ((j - 1) / (ny - 1))
    z(i, j, k) = @. z0 + (z1 - z0) * ((k - 1) / (nz - 1))

    return (x, y, z)
  end

  function initialize_mesh()
    ni = nj = nk = 7
    nhalo = 4
    x, y, z = uniform_grid(ni, nj, nk)
    mesh = CurvilinearGrid3D(x, y, z, (ni, nj, nk), nhalo)

    return mesh
  end

  mesh = initialize_mesh()

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
  @test length(scheme.b) == len
  @test length(scheme.x) == len
  @test length(scheme.solver.x) == len
  @test size(scheme.A) == (len, len)
  @test size(scheme.α) == (8, 8, 8)
  @test size(scheme.source_term) == (8, 8, 8)
  @test scheme.limits == (ilo=1, jlo=1, klo=1, ihi=8, jhi=8, khi=8)
  @test size(mesh.iterators.cell.domain) == size(scheme.iterators.domain.linear)
end
