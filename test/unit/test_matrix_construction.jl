@testset "2D ImplicitScheme matrix construction (PeriodicBC & NeumannBC}" begin
  m, n = (5, 6)

  CI = CartesianIndices((m, n))
  LI = LinearIndices(CI)
  ilo = jlo = 1
  ihi = m
  jhi = n

  bcs = (ilo=PeriodicBC(), ihi=PeriodicBC(), jlo=NeumannBC(), jhi=NeumannBC())

  # inconsistent ilo/ihi combo
  badbcs = (ilo=NeumannBC(), ihi=PeriodicBC(), jlo=NeumannBC(), jhi=NeumannBC())

  @test_throws ErrorException CurvilinearDiffusion.ImplicitSchemeType._initialize_coefficent_matrix(
    (m, n), badbcs
  )

  A = CurvilinearDiffusion.ImplicitSchemeType._initialize_coefficent_matrix((m, n), bcs)

  inner_col = LI[3, 3]
  ilo_col = LI[1, 4]
  ilo_p1_col = LI[2, 4]
  ihi_col = LI[m, 4]
  ihi_m1_col = LI[m - 1, 4] # 19

  inner_row = A[inner_col, :]
  ilo_row = A[ilo_col, :]

  ihi_row = A[ihi_col, :]

  @test ilo_row.nzind == [ilo_col, ihi_m1_col]
  @test ihi_row.nzind == [ilo_p1_col, ihi_col]

  @test inner_row.nzind == [7, 8, 9, 12, 13, 14, 17, 18, 19]

  @test length(A.nzval) == 140
end

@testset "3D ImplicitScheme matrix construction (PeriodicBC & NeumannBC}" begin
  dims = (5, 6, 7)

  CI = CartesianIndices(dims)
  LI = LinearIndices(CI)
  ilo = jlo = klo = 1
  ihi, jhi, khi = dims

  bcs = (
    ilo=PeriodicBC(),
    ihi=PeriodicBC(),
    jlo=NeumannBC(),
    jhi=NeumannBC(),
    klo=PeriodicBC(),
    khi=PeriodicBC(),
  )
  # inconsistent ilo/ihi combo
  badbcs = (
    ilo=NeumannBC(),
    ihi=PeriodicBC(),
    jlo=NeumannBC(),
    jhi=NeumannBC(),
    klo=PeriodicBC(),
    khi=PeriodicBC(),
  )

  @test_throws ErrorException CurvilinearDiffusion.ImplicitSchemeType._initialize_coefficent_matrix(
    dims, badbcs
  )

  A = CurvilinearDiffusion.ImplicitSchemeType._initialize_coefficent_matrix(dims, bcs)

  m = dims[1]
  inner_col = LI[3, 3, 3]
  ilo_col = LI[1, 4, 4]
  ilo_p1_col = LI[2, 4, 4]
  ihi_col = LI[m, 4, 4]
  ihi_m1_col = LI[m - 1, 4, 4]

  inner_row = A[inner_col, :]
  ilo_row = A[ilo_col, :]

  ihi_row = A[ihi_col, :]

  @test ilo_row.nzind == [ilo_col, ihi_m1_col]
  @test ihi_row.nzind == [ilo_p1_col, ihi_col]

  @test inner_row.nzind == [
    37,
    38,
    39,
    42,
    43,
    44,
    47,
    48,
    49,
    67,
    68,
    69,
    72,
    73,
    74,
    77,
    78,
    79,
    97,
    98,
    99,
    102,
    103,
    104,
    107,
    108,
    109,
  ]

  @test length(A.nzval) == 1864
end