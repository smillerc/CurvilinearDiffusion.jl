
@testset "1D Boundary Conditions" begin
  function uniform_grid(nx)
    x0, x1 = (-6, 6)

    x(i) = @. x0 + (x1 - x0) * ((i - 1) / (nx - 1))

    return x
  end

  function initialize_mesh()
    ni = 7
    nhalo = 4
    x = uniform_grid(ni)
    mesh = CurvilinearGrid1D(x, ni, nhalo)

    return mesh
  end

  mesh = initialize_mesh()

  domain = mesh.iterators.cell.domain

  u = zeros(cellsize_withhalo(mesh)...)

  @unpack ilo, ihi = mesh.domain_limits.cell

  for idx in domain
    u[idx] = rand()
  end

  # NeumannBC

  bcs = (ilo=NeumannBC(), ihi=NeumannBC())

  # make a copy prior to the bc application
  u_ilo = u[ilo]
  u_ihi = u[ihi]

  applybcs!(bcs, mesh, u)

  @views begin
    @test u_ilo == u[ilo - 1]
    @test u_ihi == u[ihi + 1]
  end

  # DirichletBC

  bcs = (ilo=DirichletBC(1.0), ihi=DirichletBC(2.0))

  applybcs!(bcs, mesh, u)

  @views begin
    @test all(u[ilo - 1] .== bcs.ilo.val)
    @test all(u[ihi + 1] .== bcs.ihi.val)
  end
end

@testset "2D Boundary Conditions" begin
  function uniform_grid(nx, ny)
    x0, x1 = (-6, 6)
    y0, y1 = (-6, 6)

    x(i, j) = @. x0 + (x1 - x0) * ((i - 1) / (nx - 1))
    y(i, j) = @. y0 + (y1 - y0) * ((j - 1) / (ny - 1))

    return (x, y)
  end

  function initialize_mesh()
    ni = nj = 7
    nhalo = 4
    x, y = uniform_grid(ni, nj)
    mesh = CurvilinearGrid2D(x, y, (ni, nj), nhalo)

    return mesh
  end

  mesh = initialize_mesh()

  domain = mesh.iterators.cell.domain

  u = zeros(cellsize_withhalo(mesh)...)

  @unpack ilo, ihi, jlo, jhi, = mesh.domain_limits.cell

  for idx in domain
    u[idx] = rand()
  end

  # NeumannBC

  bcs = (ilo=NeumannBC(), ihi=NeumannBC(), jlo=NeumannBC(), jhi=NeumannBC())

  # make a copy prior to the bc application
  u_ilo = u[ilo, jlo:jhi]
  u_ihi = u[ihi, jlo:jhi]
  u_jlo = u[ilo:ihi, jlo]
  u_jhi = u[ilo:ihi, jhi]

  applybcs!(bcs, mesh, u)

  @views begin
    @test u_ilo == u[ilo - 1, jlo:jhi]
    @test u_ihi == u[ihi + 1, jlo:jhi]
    @test u_jlo == u[ilo:ihi, jlo - 1]
    @test u_jhi == u[ilo:ihi, jhi + 1]
  end

  # DirichletBC

  bcs = (
    ilo=DirichletBC(1.0), ihi=DirichletBC(2.0), jlo=DirichletBC(3.0), jhi=DirichletBC(4.0)
  )

  applybcs!(bcs, mesh, u)

  @views begin
    @test all(u[ilo - 1, jlo:jhi] .== bcs.ilo.val)
    @test all(u[ihi + 1, jlo:jhi] .== bcs.ihi.val)
    @test all(u[ilo:ihi, jlo - 1] .== bcs.jlo.val)
    @test all(u[ilo:ihi, jhi + 1] .== bcs.jhi.val)
  end
end

@testset "3D Boundary Conditions" begin
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

  domain = mesh.iterators.cell.domain

  u = zeros(cellsize_withhalo(mesh)...)

  @unpack ilo, ihi, jlo, jhi, klo, khi = mesh.domain_limits.cell

  for idx in domain
    u[idx] = rand()
  end

  # NeumannBC

  bcs = (
    ilo=NeumannBC(),
    ihi=NeumannBC(),
    jlo=NeumannBC(),
    jhi=NeumannBC(),
    klo=NeumannBC(),
    khi=NeumannBC(),
  )

  # make a copy prior to the bc application
  u_ilo = u[ilo, jlo:jhi, klo:khi]
  u_ihi = u[ihi, jlo:jhi, klo:khi]
  u_jlo = u[ilo:ihi, jlo, klo:khi]
  u_jhi = u[ilo:ihi, jhi, klo:khi]
  u_klo = u[ilo:ihi, jlo:jhi, klo]
  u_khi = u[ilo:ihi, jlo:jhi, khi]

  applybcs!(bcs, mesh, u)

  @views begin
    @test u_ilo == u[ilo - 1, jlo:jhi, klo:khi]
    @test u_ihi == u[ihi + 1, jlo:jhi, klo:khi]
    @test u_jlo == u[ilo:ihi, jlo - 1, klo:khi]
    @test u_jhi == u[ilo:ihi, jhi + 1, klo:khi]
    @test u_klo == u[ilo:ihi, jlo:jhi, klo - 1]
    @test u_khi == u[ilo:ihi, jlo:jhi, khi + 1]
  end

  # DirichletBC

  bcs = (
    ilo=DirichletBC(1.0),
    ihi=DirichletBC(2.0),
    jlo=DirichletBC(3.0),
    jhi=DirichletBC(4.0),
    klo=DirichletBC(5.0),
    khi=DirichletBC(6.0),
  )

  applybcs!(bcs, mesh, u)

  @views begin
    @test all(u[ilo - 1, jlo:jhi, klo:khi] .== bcs.ilo.val)
    @test all(u[ihi + 1, jlo:jhi, klo:khi] .== bcs.ihi.val)
    @test all(u[ilo:ihi, jlo - 1, klo:khi] .== bcs.jlo.val)
    @test all(u[ilo:ihi, jhi + 1, klo:khi] .== bcs.jhi.val)
    @test all(u[ilo:ihi, jlo:jhi, klo - 1] .== bcs.klo.val)
    @test all(u[ilo:ihi, jlo:jhi, khi + 1] .== bcs.khi.val)
  end
end