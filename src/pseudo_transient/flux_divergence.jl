"""
    flux_divergence(q, mesh, idx)

Compute the divergence of the flux, e.g. ∇⋅(α∇H), where the flux is `q = α∇H`
"""
# non-conservative form
function flux_divergence(
  (qᵢ, qⱼ), u, α, cell_center_metrics, edge_metrics, idx::CartesianIndex{2}
)
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

  ∇q = ∂qᵢ∂ξ + ∂qⱼ∂η + ∂qᵢ∂η + ∂qⱼ∂ξ + ∂H∂ξ + ∂H∂η
  return ∇q
end
