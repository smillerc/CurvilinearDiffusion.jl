@inline function flux_divergence(
  (qᵢ,), cell_center_metrics, edge_metrics, idx::CartesianIndex{1}, mesh::CylindricalGrid1D
)
  i, = idx.I

  @inbounds begin
    rᵢ = CurvilinearGrids.centroid_radius(mesh, (i,))
    rᵢ₊½ = CurvilinearGrids.radius(mesh, (i + 1,))
    rᵢ₋½ = CurvilinearGrids.radius(mesh, (i,))
    ξx = cell_center_metrics.ξ.x₁[i]
    # ξxᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[i] / edge_metrics.i₊½.J[i]
    # ξxᵢ₋½ = edge_metrics.i₊½.ξ̂.x₁[i - 1] / edge_metrics.i₊½.J[i - 1]

    # aᵢⱼ = (ξx * (ξxᵢ₊½ - ξxᵢ₋½))
    # aᵢⱼ = aᵢⱼ * (abs(aᵢⱼ) > 1e-10)
    ∇q = (
      (ξx^2 / rᵢ) * (
        rᵢ₊½ * qᵢ[i] -    # q @ i+1/2
        rᵢ₋½ * qᵢ[i - 1]  # q @ i-1/2
      ) # + # ∂qᵢ∂ξ
      # aᵢⱼ * 0.5(qᵢ[i] + qᵢ[i - 1])  # ∂H∂ξ
    )
  end

  return ∇q
end

@inline function flux_divergence(
  (qᵢ,), cell_center_metrics, edge_metrics, idx::CartesianIndex{1}, mesh::SphericalGrid1D
)
  i, = idx.I # this is the cell index, not node index (important!!)

  #    h     h     d     d     d     (h=halo cell, d=domain cell)
  # 1--c--2--c--3--c--4--c--5-----6  nodes; c=centroid
  #    1     2     3                 cell indexing

  @inbounds begin
    rᵢ = CurvilinearGrids.centroid_radius(mesh, (i,))
    rᵢ₊½ = CurvilinearGrids.radius(mesh, (i + 1,))
    rᵢ₋½ = CurvilinearGrids.radius(mesh, (i,))
    ξx = cell_center_metrics.ξ.x₁[i]
    # ξxᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[i] / edge_metrics.i₊½.J[i]
    # ξxᵢ₋½ = edge_metrics.i₊½.ξ̂.x₁[i - 1] / edge_metrics.i₊½.J[i - 1]

    # aᵢⱼ = (ξx * (ξxᵢ₊½ - ξxᵢ₋½))
    # aᵢⱼ = aᵢⱼ * (abs(aᵢⱼ) > 1e-10)
    ∇q = (
      (ξx^2 / rᵢ^2) * (
        rᵢ₊½^2 * qᵢ[i] -    # q @ i+1/2
        rᵢ₋½^2 * qᵢ[i - 1]  # q @ i-1/2
      ) # + # ∂qᵢ∂ξ
      # aᵢⱼ * 0.5(qᵢ[i] + qᵢ[i - 1])  # ∂H∂ξ
    )
  end

  return ∇q
end

@inline function flux_divergence(
  (qᵢ,), cell_center_metrics, edge_metrics, idx::CartesianIndex{1}, ::CurvilinearGrid1D
)
  i, = idx.I

  @inbounds begin
    ξx = cell_center_metrics.ξ.x₁[i]
    ξxᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[i] / edge_metrics.i₊½.J[i]
    ξxᵢ₋½ = edge_metrics.i₊½.ξ̂.x₁[i - 1] / edge_metrics.i₊½.J[i - 1]

    aᵢⱼ = (ξx * (ξxᵢ₊½ - ξxᵢ₋½))
    # aᵢⱼ = aᵢⱼ * (abs(aᵢⱼ) > 1e-10)
    ∇q = (
      ξx^2 * (qᵢ[i] - qᵢ[i - 1]) + # ∂qᵢ∂ξ
      aᵢⱼ * 0.5(qᵢ[i] + qᵢ[i - 1])  # ∂H∂ξ
    )
  end

  return ∇q
end

"""
Compute the divergence of the flux, e.g. ∇⋅(α∇H), where the flux is `q = α∇H`
"""
@inline function flux_divergence(
  (qᵢ, qⱼ), cell_center_metrics, edge_metrics, idx::CartesianIndex{2}, ::CurvilinearGrid2D
)
  @inbounds begin
    i, j = idx.I

    Jᵢ₊½ = edge_metrics.i₊½.J[i, j]
    Jⱼ₊½ = edge_metrics.j₊½.J[i, j]
    Jᵢ₋½ = edge_metrics.i₊½.J[i - 1, j]
    Jⱼ₋½ = edge_metrics.j₊½.J[i, j - 1]

    ξx = cell_center_metrics.ξ.x₁[i, j]
    ξy = cell_center_metrics.ξ.x₂[i, j]
    ηx = cell_center_metrics.η.x₁[i, j]
    ηy = cell_center_metrics.η.x₂[i, j]

    ξxᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[i, j] / Jᵢ₊½
    ξyᵢ₊½ = edge_metrics.i₊½.ξ̂.x₂[i, j] / Jᵢ₊½
    ηxᵢ₊½ = edge_metrics.i₊½.η̂.x₁[i, j] / Jᵢ₊½
    ηyᵢ₊½ = edge_metrics.i₊½.η̂.x₂[i, j] / Jᵢ₊½

    ξxᵢ₋½ = edge_metrics.i₊½.ξ̂.x₁[i - 1, j] / Jᵢ₋½
    ξyᵢ₋½ = edge_metrics.i₊½.ξ̂.x₂[i - 1, j] / Jᵢ₋½
    ηxᵢ₋½ = edge_metrics.i₊½.η̂.x₁[i - 1, j] / Jᵢ₋½
    ηyᵢ₋½ = edge_metrics.i₊½.η̂.x₂[i - 1, j] / Jᵢ₋½

    ξxⱼ₊½ = edge_metrics.j₊½.ξ̂.x₁[i, j] / Jⱼ₊½
    ξyⱼ₊½ = edge_metrics.j₊½.ξ̂.x₂[i, j] / Jⱼ₊½
    ηxⱼ₊½ = edge_metrics.j₊½.η̂.x₁[i, j] / Jⱼ₊½
    ηyⱼ₊½ = edge_metrics.j₊½.η̂.x₂[i, j] / Jⱼ₊½

    ξxⱼ₋½ = edge_metrics.j₊½.ξ̂.x₁[i, j - 1] / Jⱼ₋½
    ξyⱼ₋½ = edge_metrics.j₊½.ξ̂.x₂[i, j - 1] / Jⱼ₋½
    ηxⱼ₋½ = edge_metrics.j₊½.η̂.x₁[i, j - 1] / Jⱼ₋½
    ηyⱼ₋½ = edge_metrics.j₊½.η̂.x₂[i, j - 1] / Jⱼ₋½

    # flux divergence

    aᵢⱼ = (
      ξx * (ξxᵢ₊½ - ξxᵢ₋½) +
      ξy * (ξyᵢ₊½ - ξyᵢ₋½) +
      ηx * (ξxⱼ₊½ - ξxⱼ₋½) +
      ηy * (ξyⱼ₊½ - ξyⱼ₋½)
    )

    bᵢⱼ = (
      ξx * (ηxᵢ₊½ - ηxᵢ₋½) +
      ξy * (ηyᵢ₊½ - ηyᵢ₋½) +
      ηx * (ηxⱼ₊½ - ηxⱼ₋½) +
      ηy * (ηyⱼ₊½ - ηyⱼ₋½)
    )

    ∂qᵢ∂ξ = (ξx^2 + ξy^2) * (qᵢ[i, j] - qᵢ[i - 1, j])
    ∂qⱼ∂η = (ηx^2 + ηy^2) * (qⱼ[i, j] - qⱼ[i, j - 1])

    ∂qᵢ∂η =
      0.25(ηx * ξx + ηy * ξy) * (
        (qᵢ[i, j + 1] + qᵢ[i - 1, j + 1]) - # take average on either side
        (qᵢ[i, j - 1] + qᵢ[i - 1, j - 1])   # and do diff in j
      )

    ∂qⱼ∂ξ =
      0.25(ηx * ξx + ηy * ξy) * (
        (qⱼ[i + 1, j] + qⱼ[i + 1, j - 1]) - # take average on either side
        (qⱼ[i - 1, j] + qⱼ[i - 1, j - 1])   # and do diff in i
      )

    ∂H∂ξ = aᵢⱼ * 0.5(qᵢ[i, j] + qᵢ[i - 1, j]) # ∂u/∂ξ + non-orth terms
    ∂H∂η = bᵢⱼ * 0.5(qⱼ[i, j] + qⱼ[i, j - 1]) # ∂u/∂η + non-orth terms
  end

  ∇q = ∂qᵢ∂ξ + ∂qⱼ∂η + ∂qᵢ∂η + ∂qⱼ∂ξ + ∂H∂ξ + ∂H∂η
  return ∇q
end

@inline function flux_divergence(
  (qᵢ, qⱼ),
  cell_center_metrics,
  edge_metrics,
  idx::CartesianIndex{2},
  mesh::AxisymmetricGrid2D,
)
  @inbounds begin
    i, j = idx.I

    Jᵢ₊½ = edge_metrics.i₊½.J[i, j]
    Jⱼ₊½ = edge_metrics.j₊½.J[i, j]
    Jᵢ₋½ = edge_metrics.i₊½.J[i - 1, j]
    Jⱼ₋½ = edge_metrics.j₊½.J[i, j - 1]

    ξx = cell_center_metrics.ξ.x₁[i, j]
    ξy = cell_center_metrics.ξ.x₂[i, j]
    ηx = cell_center_metrics.η.x₁[i, j]
    ηy = cell_center_metrics.η.x₂[i, j]

    ξxᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[i, j] / Jᵢ₊½
    ξyᵢ₊½ = edge_metrics.i₊½.ξ̂.x₂[i, j] / Jᵢ₊½
    ηxᵢ₊½ = edge_metrics.i₊½.η̂.x₁[i, j] / Jᵢ₊½
    ηyᵢ₊½ = edge_metrics.i₊½.η̂.x₂[i, j] / Jᵢ₊½

    ξxᵢ₋½ = edge_metrics.i₊½.ξ̂.x₁[i - 1, j] / Jᵢ₋½
    ξyᵢ₋½ = edge_metrics.i₊½.ξ̂.x₂[i - 1, j] / Jᵢ₋½
    ηxᵢ₋½ = edge_metrics.i₊½.η̂.x₁[i - 1, j] / Jᵢ₋½
    ηyᵢ₋½ = edge_metrics.i₊½.η̂.x₂[i - 1, j] / Jᵢ₋½

    ξxⱼ₊½ = edge_metrics.j₊½.ξ̂.x₁[i, j] / Jⱼ₊½
    ξyⱼ₊½ = edge_metrics.j₊½.ξ̂.x₂[i, j] / Jⱼ₊½
    ηxⱼ₊½ = edge_metrics.j₊½.η̂.x₁[i, j] / Jⱼ₊½
    ηyⱼ₊½ = edge_metrics.j₊½.η̂.x₂[i, j] / Jⱼ₊½

    ξxⱼ₋½ = edge_metrics.j₊½.ξ̂.x₁[i, j - 1] / Jⱼ₋½
    ξyⱼ₋½ = edge_metrics.j₊½.ξ̂.x₂[i, j - 1] / Jⱼ₋½
    ηxⱼ₋½ = edge_metrics.j₊½.η̂.x₁[i, j - 1] / Jⱼ₋½
    ηyⱼ₋½ = edge_metrics.j₊½.η̂.x₂[i, j - 1] / Jⱼ₋½

    # flux divergence

    aᵢⱼ = (
      ξx * (ξxᵢ₊½ - ξxᵢ₋½) +
      ξy * (ξyᵢ₊½ - ξyᵢ₋½) +
      ηx * (ξxⱼ₊½ - ξxⱼ₋½) +
      ηy * (ξyⱼ₊½ - ξyⱼ₋½)
    )

    bᵢⱼ = (
      ξx * (ηxᵢ₊½ - ηxᵢ₋½) +
      ξy * (ηyᵢ₊½ - ηyᵢ₋½) +
      ηx * (ηxⱼ₊½ - ηxⱼ₋½) +
      ηy * (ηyⱼ₊½ - ηyⱼ₋½)
    )

    r = CurvilinearGrids.centroid_radius(mesh, idx)

    ∂qᵢ∂ξ = (ξx^2 + ξy^2) * (qᵢ[i, j] - qᵢ[i - 1, j])
    ∂qⱼ∂η = (ηx^2 + ηy^2) * (qⱼ[i, j] - qⱼ[i, j - 1])

    ∂qᵢ∂η =
      0.25(ηx * ξx + ηy * ξy) * (
        (qᵢ[i, j + 1] + qᵢ[i - 1, j + 1]) - # take average on either side
        (qᵢ[i, j - 1] + qᵢ[i - 1, j - 1])   # and do diff in j
      )

    ∂qⱼ∂ξ =
      0.25(ηx * ξx + ηy * ξy) * (
        (qⱼ[i + 1, j] + qⱼ[i + 1, j - 1]) - # take average on either side
        (qⱼ[i - 1, j] + qⱼ[i - 1, j - 1])   # and do diff in i
      )

    ∂H∂ξ = aᵢⱼ * 0.5(qᵢ[i, j] + qᵢ[i - 1, j]) # ∂u/∂ξ + non-orth terms
    ∂H∂η = bᵢⱼ * 0.5(qⱼ[i, j] + qⱼ[i, j - 1]) # ∂u/∂η + non-orth terms
  end

  ∇q = ∂qᵢ∂ξ + ∂qⱼ∂η + ∂qᵢ∂η + ∂qⱼ∂ξ + ∂H∂ξ + ∂H∂η
  return ∇q
end

@inline function flux_divergence(
  (qᵢ, qⱼ, qₖ),
  cell_center_metrics,
  edge_metrics,
  idx::CartesianIndex{3},
  ::CurvilinearGrid3D,
)
  @inbounds begin
    i, j, k = idx.I

    idim, jdim, kdim = (1, 2, 3)
    ᵢ₋₁ = shift(idx, idim, -1)
    ⱼ₋₁ = shift(idx, jdim, -1)
    ₖ₋₁ = shift(idx, kdim, -1)

    Jᵢ₊½ = edge_metrics.i₊½.J[idx]
    Jⱼ₊½ = edge_metrics.j₊½.J[idx]
    Jₖ₊½ = edge_metrics.k₊½.J[idx]

    Jᵢ₋½ = edge_metrics.i₊½.J[ᵢ₋₁]
    Jⱼ₋½ = edge_metrics.j₊½.J[ⱼ₋₁]
    Jₖ₋½ = edge_metrics.k₊½.J[ₖ₋₁]

    ξx = cell_center_metrics.ξ.x₁[idx]
    ξy = cell_center_metrics.ξ.x₂[idx]
    ξz = cell_center_metrics.ξ.x₃[idx]

    ηx = cell_center_metrics.η.x₁[idx]
    ηy = cell_center_metrics.η.x₂[idx]
    ηz = cell_center_metrics.η.x₃[idx]

    ζx = cell_center_metrics.ζ.x₁[idx]
    ζy = cell_center_metrics.ζ.x₂[idx]
    ζz = cell_center_metrics.ζ.x₃[idx]

    ξxᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[idx] / Jᵢ₊½
    ξyᵢ₊½ = edge_metrics.i₊½.ξ̂.x₂[idx] / Jᵢ₊½
    ξzᵢ₊½ = edge_metrics.i₊½.ξ̂.x₃[idx] / Jᵢ₊½

    ξxᵢ₋½ = edge_metrics.i₊½.ξ̂.x₁[ᵢ₋₁] / Jᵢ₋½
    ξyᵢ₋½ = edge_metrics.i₊½.ξ̂.x₂[ᵢ₋₁] / Jᵢ₋½
    ξzᵢ₋½ = edge_metrics.i₊½.ξ̂.x₃[ᵢ₋₁] / Jᵢ₋½

    ηxᵢ₊½ = edge_metrics.i₊½.η̂.x₁[idx] / Jᵢ₊½
    ηyᵢ₊½ = edge_metrics.i₊½.η̂.x₂[idx] / Jᵢ₊½
    ηzᵢ₊½ = edge_metrics.i₊½.η̂.x₃[idx] / Jᵢ₊½

    ηxᵢ₋½ = edge_metrics.i₊½.η̂.x₁[ᵢ₋₁] / Jᵢ₋½
    ηyᵢ₋½ = edge_metrics.i₊½.η̂.x₂[ᵢ₋₁] / Jᵢ₋½
    ηzᵢ₋½ = edge_metrics.i₊½.η̂.x₃[ᵢ₋₁] / Jᵢ₋½

    ζxᵢ₊½ = edge_metrics.i₊½.ζ̂.x₁[idx] / Jᵢ₊½
    ζyᵢ₊½ = edge_metrics.i₊½.ζ̂.x₂[idx] / Jᵢ₊½
    ζzᵢ₊½ = edge_metrics.i₊½.ζ̂.x₃[idx] / Jᵢ₊½

    ζxᵢ₋½ = edge_metrics.i₊½.ζ̂.x₁[ᵢ₋₁] / Jᵢ₋½
    ζyᵢ₋½ = edge_metrics.i₊½.ζ̂.x₂[ᵢ₋₁] / Jᵢ₋½
    ζzᵢ₋½ = edge_metrics.i₊½.ζ̂.x₃[ᵢ₋₁] / Jᵢ₋½

    ξxⱼ₊½ = edge_metrics.j₊½.ξ̂.x₁[idx] / Jⱼ₊½
    ξyⱼ₊½ = edge_metrics.j₊½.ξ̂.x₂[idx] / Jⱼ₊½
    ξzⱼ₊½ = edge_metrics.j₊½.ξ̂.x₃[idx] / Jⱼ₊½

    ξxⱼ₋½ = edge_metrics.j₊½.ξ̂.x₁[ⱼ₋₁] / Jⱼ₋½
    ξyⱼ₋½ = edge_metrics.j₊½.ξ̂.x₂[ⱼ₋₁] / Jⱼ₋½
    ξzⱼ₋½ = edge_metrics.j₊½.ξ̂.x₃[ⱼ₋₁] / Jⱼ₋½

    ηxⱼ₊½ = edge_metrics.j₊½.η̂.x₁[idx] / Jⱼ₊½
    ηyⱼ₊½ = edge_metrics.j₊½.η̂.x₂[idx] / Jⱼ₊½
    ηzⱼ₊½ = edge_metrics.j₊½.η̂.x₃[idx] / Jⱼ₊½

    ηxⱼ₋½ = edge_metrics.j₊½.η̂.x₁[ⱼ₋₁] / Jⱼ₋½
    ηyⱼ₋½ = edge_metrics.j₊½.η̂.x₂[ⱼ₋₁] / Jⱼ₋½
    ηzⱼ₋½ = edge_metrics.j₊½.η̂.x₃[ⱼ₋₁] / Jⱼ₋½

    ζxⱼ₊½ = edge_metrics.j₊½.ζ̂.x₁[idx] / Jⱼ₊½
    ζyⱼ₊½ = edge_metrics.j₊½.ζ̂.x₂[idx] / Jⱼ₊½
    ζzⱼ₊½ = edge_metrics.j₊½.ζ̂.x₃[idx] / Jⱼ₊½

    ζxⱼ₋½ = edge_metrics.j₊½.ζ̂.x₁[ⱼ₋₁] / Jⱼ₋½
    ζyⱼ₋½ = edge_metrics.j₊½.ζ̂.x₂[ⱼ₋₁] / Jⱼ₋½
    ζzⱼ₋½ = edge_metrics.j₊½.ζ̂.x₃[ⱼ₋₁] / Jⱼ₋½

    ξxₖ₊½ = edge_metrics.k₊½.ξ̂.x₁[idx] / Jₖ₊½
    ξyₖ₊½ = edge_metrics.k₊½.ξ̂.x₂[idx] / Jₖ₊½
    ξzₖ₊½ = edge_metrics.k₊½.ξ̂.x₃[idx] / Jₖ₊½

    ξxₖ₋½ = edge_metrics.k₊½.ξ̂.x₁[ₖ₋₁] / Jₖ₋½
    ξyₖ₋½ = edge_metrics.k₊½.ξ̂.x₂[ₖ₋₁] / Jₖ₋½
    ξzₖ₋½ = edge_metrics.k₊½.ξ̂.x₃[ₖ₋₁] / Jₖ₋½

    ηxₖ₊½ = edge_metrics.k₊½.η̂.x₁[idx] / Jₖ₊½
    ηyₖ₊½ = edge_metrics.k₊½.η̂.x₂[idx] / Jₖ₊½
    ηzₖ₊½ = edge_metrics.k₊½.η̂.x₃[idx] / Jₖ₊½

    ηxₖ₋½ = edge_metrics.k₊½.η̂.x₁[ₖ₋₁] / Jₖ₋½
    ηyₖ₋½ = edge_metrics.k₊½.η̂.x₂[ₖ₋₁] / Jₖ₋½
    ηzₖ₋½ = edge_metrics.k₊½.η̂.x₃[ₖ₋₁] / Jₖ₋½

    ζxₖ₊½ = edge_metrics.k₊½.ζ̂.x₁[idx] / Jₖ₊½
    ζyₖ₊½ = edge_metrics.k₊½.ζ̂.x₂[idx] / Jₖ₊½
    ζzₖ₊½ = edge_metrics.k₊½.ζ̂.x₃[idx] / Jₖ₊½

    ζxₖ₋½ = edge_metrics.k₊½.ζ̂.x₁[ₖ₋₁] / Jₖ₋½
    ζyₖ₋½ = edge_metrics.k₊½.ζ̂.x₂[ₖ₋₁] / Jₖ₋½
    ζzₖ₋½ = edge_metrics.k₊½.ζ̂.x₃[ₖ₋₁] / Jₖ₋½

    # flux divergence

    αᵢⱼₖ = (
      ξx * (ξxᵢ₊½ - ξxᵢ₋½) +
      ξy * (ξyᵢ₊½ - ξyᵢ₋½) +
      ξz * (ξzᵢ₊½ - ξzᵢ₋½) +
      #
      ηx * (ξxⱼ₊½ - ξxⱼ₋½) +
      ηy * (ξyⱼ₊½ - ξyⱼ₋½) +
      ηz * (ξzⱼ₊½ - ξzⱼ₋½) +
      #
      ζx * (ξxₖ₊½ - ξxₖ₋½) +
      ζy * (ξyₖ₊½ - ξyₖ₋½) +
      ζz * (ξzₖ₊½ - ξzₖ₋½)
    )

    βᵢⱼₖ = (
      ξx * (ηxᵢ₊½ - ηxᵢ₋½) +
      ξy * (ηyᵢ₊½ - ηyᵢ₋½) +
      ξz * (ηzᵢ₊½ - ηzᵢ₋½) +
      #
      ηx * (ηxⱼ₊½ - ηxⱼ₋½) +
      ηy * (ηyⱼ₊½ - ηyⱼ₋½) +
      ηz * (ηzⱼ₊½ - ηzⱼ₋½) +
      #
      ζx * (ηxₖ₊½ - ηxₖ₋½) +
      ζy * (ηyₖ₊½ - ηyₖ₋½) +
      ζz * (ηzₖ₊½ - ηzₖ₋½)
    )

    γᵢⱼₖ = (
      ξx * (ζxᵢ₊½ - ζxᵢ₋½) +
      ξy * (ζyᵢ₊½ - ζyᵢ₋½) +
      ξz * (ζzᵢ₊½ - ζzᵢ₋½) +
      #
      ηx * (ζxⱼ₊½ - ζxⱼ₋½) +
      ηy * (ζyⱼ₊½ - ζyⱼ₋½) +
      ηz * (ζzⱼ₊½ - ζzⱼ₋½) +
      #
      ζx * (ζxₖ₊½ - ζxₖ₋½) +
      ζy * (ζyₖ₊½ - ζyₖ₋½) +
      ζz * (ζzₖ₊½ - ζzₖ₋½)
    )

    ∂qᵢ∂ξ = (ξx^2 + ξy^2 + ξz^2) * (qᵢ[i, j, k] - qᵢ[i - 1, j, k])
    ∂qⱼ∂η = (ηx^2 + ηy^2 + ηz^2) * (qⱼ[i, j, k] - qⱼ[i, j - 1, k])
    ∂qₖ∂ζ = (ζx^2 + ζy^2 + ζz^2) * (qₖ[i, j, k] - qₖ[i, j, k - 1])

    # ---------------
    # ∂/∂η (α ∂u/∂ξ), aka ∂qᵢ/∂η
    # inner index is i  , i-1
    # outer index is j-1, j+1
    ∂qᵢ∂η =
      0.5(ηx * ξx + ηy * ξy + ηz * ξz) * (
        #  take average and do diff in j (for ∂/∂η)
        0.5(qᵢ[i, j + 1, k] + qᵢ[i - 1, j + 1, k]) - # j + 1
        0.5(qᵢ[i, j - 1, k] + qᵢ[i - 1, j - 1, k])   # j - 1
      )

    # ∂/∂ξ (α ∂u/∂η), aka ∂qⱼ/∂ξ
    # inner index is j  , j-1
    # outer index is i-1, i+1
    ∂qⱼ∂ξ =
      0.5(ηx * ξx + ηy * ξy + ηz * ξz) * (
        #  take average and do diff in i (for ∂/∂ξ)
        0.5(qⱼ[i + 1, j, k] + qⱼ[i + 1, j - 1, k]) - # i + 1
        0.5(qⱼ[i - 1, j, k] + qⱼ[i - 1, j - 1, k])   # i - 1
      )

    # # ---------------

    # ∂/∂ζ (α ∂u/∂η), aka ∂qⱼ/∂ζ
    # inner index is j  , j-1
    # outer index is k-1, k+1
    ∂qⱼ∂ζ =
      0.5(ζx * ηx + ζy * ηy + ζz * ηz) * (
        #  take average and do diff in k (for ∂/∂ζ)
        0.5(qⱼ[i, j, k + 1] + qⱼ[i, j - 1, k + 1]) - # k + 1
        0.5(qⱼ[i, j, k - 1] + qⱼ[i, j - 1, k - 1])   # k - 1
      )

    # ∂/∂η (α ∂u/∂ζ), aka ∂qₖ/∂η
    # inner index is k  , k-1
    # outer index is j-1, j+1
    ∂qₖ∂η =
      0.5(ζx * ηx + ζy * ηy + ζz * ηz) * (
        #  take average and do diff in j (for ∂/∂η)
        0.5(qₖ[i, j + 1, k] + qₖ[i, j + 1, k - 1]) - # j + 1
        0.5(qₖ[i, j - 1, k] + qₖ[i, j - 1, k - 1])   # j - 1
      )

    # # ---------------

    # ∂/∂ζ (α ∂u/∂ξ), aka ∂qᵢ/∂ζ
    # inner index is i  , i-1
    # outer index is k-1, k+1
    ∂qᵢ∂ζ =
      0.5(ζx * ξx + ζy * ξy + ζz * ξz) * (
        #  take average and do diff in k (for ∂/∂ζ)
        0.5(qᵢ[i, j, k + 1] + qᵢ[i - 1, j, k + 1]) - # k + 1
        0.5(qᵢ[i, j, k - 1] + qᵢ[i - 1, j, k - 1])   # k - 1
      )

    # ∂/∂ξ (α ∂u/∂ζ), aka ∂qₖ/∂ξ
    # inner index is k  , k-1
    # outer index is i-1, i+1
    ∂qₖ∂ξ =
      0.5(ζx * ξx + ζy * ξy + ζz * ξz) * (
        #  take average and do diff in i (for ∂/∂ξ)
        0.5(qₖ[i + 1, j, k] + qₖ[i + 1, j, k - 1]) - # i + 1
        0.5(qₖ[i - 1, j, k] + qₖ[i - 1, j, k - 1])   # i - 1
      )

    # ---------------

    # additional non-orthogonal terms
    ∂q∂ξ_α = αᵢⱼₖ * 0.5(qᵢ[i, j, k] + qᵢ[i - 1, j, k])
    ∂q∂η_β = βᵢⱼₖ * 0.5(qⱼ[i, j, k] + qⱼ[i, j - 1, k])
    ∂q∂ζ_γ = γᵢⱼₖ * 0.5(qₖ[i, j, k] + qₖ[i, j, k - 1])
  end

  ∇q = (
    ∂qᵢ∂ξ +
    ∂qⱼ∂η +
    ∂qₖ∂ζ +
    #
    ∂qᵢ∂η +
    ∂qᵢ∂ζ +
    ∂qⱼ∂ξ +
    ∂qⱼ∂ζ +
    ∂qₖ∂η +
    ∂qₖ∂ξ +
    #
    ∂q∂ξ_α +
    ∂q∂η_β +
    ∂q∂ζ_γ
  )
  return ∇q
end
