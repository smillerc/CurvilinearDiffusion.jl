
@testset "1D Boundary Conditions" begin
  x0, x1 = (-6, 6)
  ni = 7
  nhalo = 4
  mesh = RectlinearGrid(x0, x1, ni, nhalo)
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
  x0, x1 = (-6, 6)
  y0, y1 = (-6, 6)
  ni = nj = 6
  nhalo = 4
  mesh = RectlinearGrid((x0, y0), (x1, y1), (ni, nj), nhalo)

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
  x0, x1 = (-6, 6)
  y0, y1 = (-6, 6)
  z0, z1 = (-6, 6)
  ni = nj = nk = 6
  nhalo = 4

  mesh = RectlinearGrid((x0, y0, z0), (x1, y1, z1), (ni, nj, nk), nhalo)

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