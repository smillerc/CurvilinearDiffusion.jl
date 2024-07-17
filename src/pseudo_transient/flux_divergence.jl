"""
    flux_divergence(qH, mesh, idx)

Compute the divergence of the flux, e.g. ∇⋅(α∇H), where the flux is `qH = α∇H`
"""
function flux_divergence_cons(
  (qᵢ, qⱼ), H, α, cell_center_metrics, edge_metrics, idx::CartesianIndex{2}
)
  i, j = idx.I

  # αᵢ₊½ = 0.5(α[i, j] + α[i + 1, j])
  # αᵢ₋½ = 0.5(α[i, j] + α[i - 1, j])
  # αⱼ₊½ = 0.5(α[i, j] + α[i, j + 1])
  # αⱼ₋½ = 0.5(α[i, j] + α[i, j - 1])

  # Jᵢ₊½ = edge_metrics.i₊½.J[i, j]
  # Jⱼ₊½ = edge_metrics.j₊½.J[i, j]
  # Jᵢ₋½ = edge_metrics.i₊½.J[i - 1, j]
  # Jⱼ₋½ = edge_metrics.j₊½.J[i, j - 1]

  # Jξx_ᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[i, j]
  # Jξy_ᵢ₊½ = edge_metrics.i₊½.ξ̂.x₂[i, j]
  # Jηx_ᵢ₊½ = edge_metrics.i₊½.η̂.x₁[i, j]
  # Jηy_ᵢ₊½ = edge_metrics.i₊½.η̂.x₂[i, j]

  # Jξx_ᵢ₋½ = edge_metrics.i₊½.ξ̂.x₁[i - 1, j]
  # Jξy_ᵢ₋½ = edge_metrics.i₊½.ξ̂.x₂[i - 1, j]
  # Jηx_ᵢ₋½ = edge_metrics.i₊½.η̂.x₁[i - 1, j]
  # Jηy_ᵢ₋½ = edge_metrics.i₊½.η̂.x₂[i - 1, j]

  # Jξx_ⱼ₊½ = edge_metrics.j₊½.ξ̂.x₁[i, j]
  # Jξy_ⱼ₊½ = edge_metrics.j₊½.ξ̂.x₂[i, j]
  # Jηx_ⱼ₊½ = edge_metrics.j₊½.η̂.x₁[i, j]
  # Jηy_ⱼ₊½ = edge_metrics.j₊½.η̂.x₂[i, j]

  # Jξx_ⱼ₋½ = edge_metrics.j₊½.ξ̂.x₁[i, j - 1]
  # Jξy_ⱼ₋½ = edge_metrics.j₊½.ξ̂.x₂[i, j - 1]
  # Jηx_ⱼ₋½ = edge_metrics.j₊½.η̂.x₁[i, j - 1]
  # Jηy_ⱼ₋½ = edge_metrics.j₊½.η̂.x₂[i, j - 1]

  # a_Jξ²ᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
  # a_Jξ²ᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2) / Jᵢ₋½
  # a_Jη²ⱼ₊½ = αⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½
  # a_Jη²ⱼ₋½ = αⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2) / Jⱼ₋½

  # a_Jξηᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½ * Jηx_ᵢ₊½ + Jξy_ᵢ₊½ * Jηy_ᵢ₊½) / (4Jᵢ₊½)
  # a_Jξηᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½ * Jηx_ᵢ₋½ + Jξy_ᵢ₋½ * Jηy_ᵢ₋½) / (4Jᵢ₋½)
  # a_Jηξⱼ₊½ = αⱼ₊½ * (Jξx_ⱼ₊½ * Jηx_ⱼ₊½ + Jξy_ⱼ₊½ * Jηy_ⱼ₊½) / (4Jⱼ₊½)
  # a_Jηξⱼ₋½ = αⱼ₋½ * (Jξx_ⱼ₋½ * Jηx_ⱼ₋½ + Jξy_ⱼ₋½ * Jηy_ⱼ₋½) / (4Jⱼ₋½)

  # flux divergence

  # ∇q = (
  #   (a_Jξ²ᵢ₊½ * (H[i + 1, j] - H[i, j]) - a_Jξ²ᵢ₋½ * (H[i, j] - H[i - 1, j])) +
  #   (a_Jη²ⱼ₊½ * (H[i, j + 1] - H[i, j]) - a_Jη²ⱼ₋½ * (H[i, j] - H[i, j - 1]))
  #   # +
  #   # a_Jξηᵢ₊½ * (H[i, j + 1] - H[i, j - 1] + H[i + 1, j + 1] - H[i + 1, j - 1]) -
  #   # a_Jξηᵢ₋½ * (H[i, j + 1] - H[i, j - 1] + H[i - 1, j + 1] - H[i - 1, j - 1]) +
  #   # a_Jηξⱼ₊½ * (H[i + 1, j] - H[i - 1, j] + H[i + 1, j + 1] - H[i - 1, j + 1]) -
  #   # a_Jηξⱼ₋½ * (H[i + 1, j] - H[i - 1, j] + H[i + 1, j - 1] - H[i - 1, j - 1])
  # )
  ∇q = (
    (qᵢ[i, j] - qᵢ[i - 1, j]) + # 
    (qⱼ[i, j] - qⱼ[i, j - 1])
    # + # 
    # a_Jξηᵢ₊½ * (H[i, j + 1] - H[i, j - 1] + H[i + 1, j + 1] - H[i + 1, j - 1]) -
    # a_Jξηᵢ₋½ * (H[i, j + 1] - H[i, j - 1] + H[i - 1, j + 1] - H[i - 1, j - 1]) +
    # a_Jηξⱼ₊½ * (H[i + 1, j] - H[i - 1, j] + H[i + 1, j + 1] - H[i - 1, j + 1]) -
    # a_Jηξⱼ₋½ * (H[i + 1, j] - H[i - 1, j] + H[i + 1, j - 1] - H[i - 1, j - 1])
  )

  # ∇q = ∂qᵢ∂ξ + ∂qⱼ∂η + ∂qᵢ∂η + ∂qⱼ∂ξ + ∂H∂ξ + ∂H∂η
  return ∇q
end

# non-conservative form
function flux_divergence(
  (qᵢ, qⱼ), H, α, cell_center_metrics, edge_metrics, idx::CartesianIndex{2}
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

  # ∂qᵢ∂η =
  #   0.25(ηx * ξx + ηy * ξy) * (
  #     α[i + 1, j] * (H[i + 1, j + 1] - H[i + 1, j - 1]) -
  #     α[i - 1, j] * (H[i - 1, j + 1] - H[i - 1, j - 1])
  #   )

  # ∂qⱼ∂ξ =
  #   0.25(ηx * ξx + ηy * ξy) * (
  #     α[i, j + 1] * (H[i + 1, j + 1] - H[i - 1, j + 1]) -
  #     α[i, j - 1] * (H[i + 1, j - 1] - H[i - 1, j - 1])
  #   )

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
  # ∂H∂ξ = 0.5α[i, j] * aᵢⱼ * (H[i + 1, j] - H[i - 1, j])  # ∂H/∂ξ + non-orth terms
  # ∂H∂η = 0.5α[i, j] * bᵢⱼ * (H[i, j + 1] - H[i, j - 1])  # ∂H/∂η + non-orth terms

  # ∂qᵢ∂ξ = ∂qᵢ∂ξ * (abs(∂qᵢ∂ξ) <= 1e-14)
  # ∂qⱼ∂η = ∂qⱼ∂η * (abs(∂qⱼ∂η) <= 1e-14)
  # ∂qᵢ∂η = ∂qᵢ∂η * (abs(∂qᵢ∂η) <= 1e-14)
  # ∂qⱼ∂ξ = ∂qⱼ∂ξ * (abs(∂qⱼ∂ξ) <= 1e-14)
  # ∂H∂ξ = ∂H∂ξ * (abs(∂H∂ξ) <= 1e-14)
  # ∂H∂η = ∂H∂η * (abs(∂H∂η) <= 1e-14)

  ∇q = ∂qᵢ∂ξ + ∂qⱼ∂η + ∂qᵢ∂η + ∂qⱼ∂ξ + ∂H∂ξ + ∂H∂η
  return ∇q
end

function flux_divergence_orig(
  (qHx, qHy), H, α, cell_center_metrics, edge_metrics, idx::CartesianIndex{2}
)
  m = non_conservative_metrics(cell_center_metrics, edge_metrics, idx)

  iaxis = 1
  jaxis = 2
  ᵢ₋₁ = shift(idx, iaxis, -1)
  ⱼ₋₁ = shift(idx, jaxis, -1)

  αᵢⱼ = (
    m.ξx * (m.ξxᵢ₊½ - m.ξxᵢ₋½) +
    m.ξy * (m.ξyᵢ₊½ - m.ξyᵢ₋½) +
    m.ηx * (m.ξxⱼ₊½ - m.ξxⱼ₋½) +
    m.ηy * (m.ξyⱼ₊½ - m.ξyⱼ₋½)
  )

  βᵢⱼ = (
    m.ξx * (m.ηxᵢ₊½ - m.ηxᵢ₋½) +
    m.ξy * (m.ηyᵢ₊½ - m.ηyᵢ₋½) +
    m.ηx * (m.ηxⱼ₊½ - m.ηxⱼ₋½) +
    m.ηy * (m.ηyⱼ₊½ - m.ηyⱼ₋½)
  )

  ∇qH = (
    # (qHx[idx] - qHx[ᵢ₋₁]) + # ∂qH∂x
    # (qHy[idx] - qHy[ⱼ₋₁])   # ∂qH∂y
    (m.ξx^2 + m.ξy^2) * (qHx[idx] - qHx[ᵢ₋₁]) + # ∂qH∂x
    (m.ηx^2 + m.ηy^2) * (qHy[idx] - qHy[ⱼ₋₁])   # ∂qH∂y
  )

  return ∇qH
end