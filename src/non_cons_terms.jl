
using CartesianDomains

@inline function non_cons_terms(cell_center_metrics, edge_metrics, idx::CartesianIndex{1})
  ᵢ₋₁ = shift(idx, 1, -1)

  # The conserved metrics are stored at the edges, but we 
  # want the "non-conservative" version
  Jᵢ₊½ = edge_metrics.i₊½.J[idx]
  Jᵢ₋½ = edge_metrics.i₋½.J[ᵢ₋₁]

  ξxᵢ₋½ = edge_metrics.j₋½.ξ̂.x₁[ᵢ₋₁] / Jᵢ₋½
  ξxᵢ₊½ = edge_metrics.j₊½.ξ̂.x₁[idx] / Jᵢ₊½
  ξx = cell_center_metrics.ξ.x₁[idx]

  αᵢⱼ = (ξx * (ξxᵢ₊½ - ξxᵢ₋½))

  f_ξ = ξx^2

  return (; α=αᵢⱼ, f_ξ)
end

@inline function non_cons_terms(cell_center_metrics, edge_metrics, idx::CartesianIndex{2})
  idim, jdim = (1, 2)
  ᵢ₋₁ = shift(idx, idim, -1)
  ⱼ₋₁ = shift(idx, jdim, -1)

  # The conserved metrics are stored at the edges, but we 
  # want the "non-conservative" version
  Jᵢ₊½ = edge_metrics.i₊½.J[idx]
  Jᵢ₋½ = edge_metrics.i₋½.J[ᵢ₋₁]
  Jⱼ₊½ = edge_metrics.j₊½.J[idx]
  Jⱼ₋½ = edge_metrics.j₋½.J[ⱼ₋₁]

  ξxᵢ₋½ = edge_metrics.i₋½.ξ̂.x₁[ᵢ₋₁] / Jᵢ₋½
  ξyᵢ₋½ = edge_metrics.i₋½.ξ̂.x₂[ᵢ₋₁] / Jᵢ₋½
  ηxᵢ₋½ = edge_metrics.i₋½.η̂.x₁[ᵢ₋₁] / Jᵢ₋½
  ηyᵢ₋½ = edge_metrics.i₋½.η̂.x₂[ᵢ₋₁] / Jᵢ₋½

  ξxⱼ₋½ = edge_metrics.j₋½.ξ̂.x₁[ⱼ₋₁] / Jⱼ₋½
  ξyⱼ₋½ = edge_metrics.j₋½.ξ̂.x₂[ⱼ₋₁] / Jⱼ₋½
  ηxⱼ₋½ = edge_metrics.j₋½.η̂.x₁[ⱼ₋₁] / Jⱼ₋½
  ηyⱼ₋½ = edge_metrics.j₋½.η̂.x₂[ⱼ₋₁] / Jⱼ₋½

  ξxᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[idx] / Jᵢ₊½
  ξyᵢ₊½ = edge_metrics.i₊½.ξ̂.x₂[idx] / Jᵢ₊½
  ηxᵢ₊½ = edge_metrics.i₊½.η̂.x₁[idx] / Jᵢ₊½
  ηyᵢ₊½ = edge_metrics.i₊½.η̂.x₂[idx] / Jᵢ₊½

  ξxⱼ₊½ = edge_metrics.j₊½.ξ̂.x₁[idx] / Jⱼ₊½
  ξyⱼ₊½ = edge_metrics.j₊½.ξ̂.x₂[idx] / Jⱼ₊½
  ηxⱼ₊½ = edge_metrics.j₊½.η̂.x₁[idx] / Jⱼ₊½
  ηyⱼ₊½ = edge_metrics.j₊½.η̂.x₂[idx] / Jⱼ₊½

  ξx = cell_center_metrics.ξ.x₁[idx]
  ξy = cell_center_metrics.ξ.x₂[idx]
  ηx = cell_center_metrics.η.x₁[idx]
  ηy = cell_center_metrics.η.x₂[idx]

  #! format: off
  αᵢⱼ = (
    ξx * (ξxᵢ₊½ - ξxᵢ₋½) + ξy * (ξyᵢ₊½ - ξyᵢ₋½) + 
    ηx * (ξxⱼ₊½ - ξxⱼ₋½) + ηy * (ξyⱼ₊½ - ξyⱼ₋½)
  )

  βᵢⱼ = (
    ξx * (ηxᵢ₊½ - ηxᵢ₋½) + ξy * (ηyᵢ₊½ - ηyᵢ₋½) +
    ηx * (ηxⱼ₊½ - ηxⱼ₋½) + ηy * (ηyⱼ₊½ - ηyⱼ₋½)
  )
  #! format: on

  f_ξ = (ξx^2 + ξy^2)
  f_η = (ηx^2 + ηy^2)
  f_ξη = (ξx * ηx + ξy * ηy) / 4

  return (; α=αᵢⱼ, β=βᵢⱼ, f_ξ, f_η, f_ξη)
end

@inline function non_cons_terms(cell_center_metrics, edge_metrics, idx::CartesianIndex{3})
  idim, jdim, kdim = (1, 2, 3)
  ᵢ₋₁ = shift(idx, idim, -1)
  ⱼ₋₁ = shift(idx, jdim, -1)
  ₖ₋₁ = shift(idx, kdim, -1)

  # The conserved metrics are stored at the edges, but we 
  # want the "non-conservative" version
  Jᵢ₊½ = edge_metrics.i₊½.J[idx]
  Jᵢ₋½ = edge_metrics.i₋½.J[ᵢ₋₁]

  Jⱼ₊½ = edge_metrics.j₊½.J[idx]
  Jⱼ₋½ = edge_metrics.j₋½.J[ⱼ₋₁]

  Jₖ₊½ = edge_metrics.k₊½.J[idx]
  Jₖ₋½ = edge_metrics.k₋½.J[ₖ₋₁]

  #ᵢ₋½
  ξxᵢ₋½ = edge_metrics.i₋½.ξ̂.x₁[ᵢ₋₁] / Jᵢ₋½
  ξyᵢ₋½ = edge_metrics.i₋½.ξ̂.x₂[ᵢ₋₁] / Jᵢ₋½
  ξzᵢ₋½ = edge_metrics.i₋½.ξ̂.x₃[ᵢ₋₁] / Jᵢ₋½
  ηxᵢ₋½ = edge_metrics.i₋½.η̂.x₁[ᵢ₋₁] / Jᵢ₋½
  ηyᵢ₋½ = edge_metrics.i₋½.η̂.x₂[ᵢ₋₁] / Jᵢ₋½
  ηzᵢ₋½ = edge_metrics.i₋½.η̂.x₃[ᵢ₋₁] / Jᵢ₋½
  ζxᵢ₋½ = edge_metrics.i₋½.ζ̂.x₁[ᵢ₋₁] / Jᵢ₋½
  ζyᵢ₋½ = edge_metrics.i₋½.ζ̂.x₂[ᵢ₋₁] / Jᵢ₋½
  ζzᵢ₋½ = edge_metrics.i₋½.ζ̂.x₃[ᵢ₋₁] / Jᵢ₋½

  ∂x/∂ξ
  ∂ξ/∂x
  
  # j₋½
  ξxⱼ₋½ = edge_metrics.j₋½.ξ̂.x₁[ⱼ₋₁] / Jⱼ₋½
  ξyⱼ₋½ = edge_metrics.j₋½.ξ̂.x₂[ⱼ₋₁] / Jⱼ₋½
  ξzⱼ₋½ = edge_metrics.j₋½.ξ̂.x₃[ⱼ₋₁] / Jⱼ₋½
  ηxⱼ₋½ = edge_metrics.j₋½.η̂.x₁[ⱼ₋₁] / Jⱼ₋½
  ηyⱼ₋½ = edge_metrics.j₋½.η̂.x₂[ⱼ₋₁] / Jⱼ₋½
  ηzⱼ₋½ = edge_metrics.j₋½.η̂.x₃[ⱼ₋₁] / Jⱼ₋½
  ζxⱼ₋½ = edge_metrics.j₋½.ζ̂.x₁[ⱼ₋₁] / Jⱼ₋½
  ζyⱼ₋½ = edge_metrics.j₋½.ζ̂.x₂[ⱼ₋₁] / Jⱼ₋½
  ζzⱼ₋½ = edge_metrics.j₋½.ζ̂.x₃[ⱼ₋₁] / Jⱼ₋½

  # k₋½
  ξxₖ₋½ = edge_metrics.k₋½.ξ̂.x₁[ₖ₋₁] / Jₖ₋½
  ξyₖ₋½ = edge_metrics.k₋½.ξ̂.x₂[ₖ₋₁] / Jₖ₋½
  ξzₖ₋½ = edge_metrics.k₋½.ξ̂.x₃[ₖ₋₁] / Jₖ₋½
  ηxₖ₋½ = edge_metrics.k₋½.η̂.x₁[ₖ₋₁] / Jₖ₋½
  ηyₖ₋½ = edge_metrics.k₋½.η̂.x₂[ₖ₋₁] / Jₖ₋½
  ηzₖ₋½ = edge_metrics.k₋½.η̂.x₃[ₖ₋₁] / Jₖ₋½
  ζxₖ₋½ = edge_metrics.k₋½.ζ̂.x₁[ₖ₋₁] / Jₖ₋½
  ζyₖ₋½ = edge_metrics.k₋½.ζ̂.x₂[ₖ₋₁] / Jₖ₋½
  ζzₖ₋½ = edge_metrics.k₋½.ζ̂.x₃[ₖ₋₁] / Jₖ₋½

  # i₊½
  ξxᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[idx] / Jᵢ₊½
  ξyᵢ₊½ = edge_metrics.i₊½.ξ̂.x₂[idx] / Jᵢ₊½
  ξzᵢ₊½ = edge_metrics.i₊½.ξ̂.x₃[idx] / Jᵢ₊½
  ηxᵢ₊½ = edge_metrics.i₊½.η̂.x₁[idx] / Jᵢ₊½
  ηyᵢ₊½ = edge_metrics.i₊½.η̂.x₂[idx] / Jᵢ₊½
  ηzᵢ₊½ = edge_metrics.i₊½.η̂.x₃[idx] / Jᵢ₊½
  ζxᵢ₊½ = edge_metrics.i₊½.ζ̂.x₁[idx] / Jᵢ₊½
  ζyᵢ₊½ = edge_metrics.i₊½.ζ̂.x₂[idx] / Jᵢ₊½
  ζzᵢ₊½ = edge_metrics.i₊½.ζ̂.x₃[idx] / Jᵢ₊½

  # j₊½
  ξxⱼ₊½ = edge_metrics.j₊½.ξ̂.x₁[idx] / Jⱼ₊½
  ξyⱼ₊½ = edge_metrics.j₊½.ξ̂.x₂[idx] / Jⱼ₊½
  ξzⱼ₊½ = edge_metrics.j₊½.ξ̂.x₃[idx] / Jⱼ₊½
  ηxⱼ₊½ = edge_metrics.j₊½.η̂.x₁[idx] / Jⱼ₊½
  ηyⱼ₊½ = edge_metrics.j₊½.η̂.x₂[idx] / Jⱼ₊½
  ηzⱼ₊½ = edge_metrics.j₊½.η̂.x₃[idx] / Jⱼ₊½
  ζxⱼ₊½ = edge_metrics.j₊½.ζ̂.x₁[idx] / Jⱼ₊½
  ζyⱼ₊½ = edge_metrics.j₊½.ζ̂.x₂[idx] / Jⱼ₊½
  ζzⱼ₊½ = edge_metrics.j₊½.ζ̂.x₃[idx] / Jⱼ₊½

  # k₊½
  ξxₖ₊½ = edge_metrics.k₊½.ξ̂.x₁[idx] / Jₖ₊½
  ξyₖ₊½ = edge_metrics.k₊½.ξ̂.x₂[idx] / Jₖ₊½
  ξzₖ₊½ = edge_metrics.k₊½.ξ̂.x₃[idx] / Jₖ₊½
  ηxₖ₊½ = edge_metrics.k₊½.η̂.x₁[idx] / Jₖ₊½
  ηyₖ₊½ = edge_metrics.k₊½.η̂.x₂[idx] / Jₖ₊½
  ηzₖ₊½ = edge_metrics.k₊½.η̂.x₃[idx] / Jₖ₊½
  ζxₖ₊½ = edge_metrics.k₊½.ζ̂.x₁[idx] / Jₖ₊½
  ζyₖ₊½ = edge_metrics.k₊½.ζ̂.x₂[idx] / Jₖ₊½
  ζzₖ₊½ = edge_metrics.k₊½.ζ̂.x₃[idx] / Jₖ₊½

  ξx = cell_center_metrics.ξ.x₁[idx]
  ξy = cell_center_metrics.ξ.x₂[idx]
  ξz = cell_center_metrics.ξ.x₃[idx]

  ηx = cell_center_metrics.η.x₁[idx]
  ηy = cell_center_metrics.η.x₂[idx]
  ηz = cell_center_metrics.η.x₃[idx]

  ζx = cell_center_metrics.ζ.x₁[idx]
  ζy = cell_center_metrics.ζ.x₂[idx]
  ζz = cell_center_metrics.ζ.x₃[idx]

  #! format: off
  αᵢⱼₖ = (
    ξx * (ξxᵢ₊½ - ξxᵢ₋½) + ξy * (ξyᵢ₊½ - ξyᵢ₋½) + ξz * (ξzᵢ₊½ - ξzᵢ₋½) +
    ηx * (ξxⱼ₊½ - ξxⱼ₋½) + ηy * (ξyⱼ₊½ - ξyⱼ₋½) + ηz * (ξzⱼ₊½ - ξzⱼ₋½) +
    ζx * (ξxₖ₊½ - ξxₖ₋½) + ζy * (ξyₖ₊½ - ξyₖ₋½) + ζz * (ξzₖ₊½ - ξzₖ₋½)
  )

  βᵢⱼₖ = (
    ξx * (ηxᵢ₊½ - ηxᵢ₋½) + ξy * (ηyᵢ₊½ - ηyᵢ₋½) + ξz * (ηzᵢ₊½ - ηzᵢ₋½) +
    ηx * (ηxⱼ₊½ - ηxⱼ₋½) + ηy * (ηyⱼ₊½ - ηyⱼ₋½) + ηz * (ηzⱼ₊½ - ηzⱼ₋½) +
    ζx * (ηxₖ₊½ - ηxₖ₋½) + ζy * (ηyₖ₊½ - ηyₖ₋½) + ζz * (ηzₖ₊½ - ηzₖ₋½)
  )

  γᵢⱼₖ = (
    ξx * (ζxᵢ₊½ - ζxᵢ₋½) + ξy * (ζyᵢ₊½ - ζyᵢ₋½) + ξz * (ζzᵢ₊½ - ζzᵢ₋½) +
    ηx * (ζxⱼ₊½ - ζxⱼ₋½) + ηy * (ζyⱼ₊½ - ζyⱼ₋½) + ηz * (ζzⱼ₊½ - ζzⱼ₋½) +
    ζx * (ζxₖ₊½ - ζxₖ₋½) + ζy * (ζyₖ₊½ - ζyₖ₋½) + ζz * (ζzₖ₊½ - ζzₖ₋½)
  )
  #! format: on

  f_ξ = ξx^2 + ξy^2 + ξz^2
  f_η = ηx^2 + ηy^2 + ηz^2
  f_ζ = ζx^2 + ζy^2 + ζz^2
  f_ξηζ = (
    ξx * ηx * ζy +  #
    ξy * ηy * ζy +  #
    ξz * ηz * ζz    #
  ) / ??4

  return (; α=αᵢⱼₖ, β=βᵢⱼₖ, γ=γᵢⱼₖ, f_ξ, f_η, f_ζ, f_ξηζ)
end
