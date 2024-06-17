
using CartesianDomains

"""
  conservative_edge_terms(edge_diffusivity::NTuple{4,T}, m) where {T}

Collect and find the 2D edge terms used for the conservative form of the diffusion equation.
These are essentially the edge diffusivity + grid metric terms
"""
@inline function conservative_edge_terms(
  edge_diffusivity, edge_metrics, idx::CartesianIndex{1}
)
  @unpack αᵢ₊½, αᵢ₋½ = edge_diffusivity

  ᵢ₋₁ = idx.I - 1

  Jξx_ᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[idx]
  Jξx_ᵢ₋½ = edge_metrics.i₊½.ξ̂.x₁[ᵢ₋₁]

  fᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½^2) / ᵢ₊½.J
  fᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½^2) / ᵢ₋½.J

  return (; fᵢ₊½, fᵢ₋½)
end

"""
  conservative_edge_terms(edge_diffusivity::NTuple{4,T}, m) where {T}

Collect and find the 2D edge terms used for the conservative form of the diffusion equation.
These are essentially the edge diffusivity + grid metric terms
"""
@inline function conservative_edge_terms(
  edge_diffusivity, edge_metrics, idx::CartesianIndex{2}
)
  @unpack αᵢ₊½, αᵢ₋½, αⱼ₊½, αⱼ₋½ = edge_diffusivity

  idim, jdim = (1, 2)
  ᵢ₋₁ = shift(idx, idim, -1)
  ⱼ₋₁ = shift(idx, jdim, -1)

  Jᵢ₊½ = edge_metrics.i₊½.J[idx]
  Jⱼ₊½ = edge_metrics.j₊½.J[idx]
  Jᵢ₋½ = edge_metrics.i₊½.J[ᵢ₋₁]
  Jⱼ₋½ = edge_metrics.j₊½.J[ⱼ₋₁]

  Jξx_ᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[idx]
  Jξy_ᵢ₊½ = edge_metrics.i₊½.ξ̂.x₂[idx]
  Jηx_ᵢ₊½ = edge_metrics.i₊½.η̂.x₁[idx]
  Jηy_ᵢ₊½ = edge_metrics.i₊½.η̂.x₂[idx]

  Jξx_ᵢ₋½ = edge_metrics.i₊½.ξ̂.x₁[ᵢ₋₁]
  Jξy_ᵢ₋½ = edge_metrics.i₊½.ξ̂.x₂[ᵢ₋₁]
  Jηx_ᵢ₋½ = edge_metrics.i₊½.η̂.x₁[ᵢ₋₁]
  Jηy_ᵢ₋½ = edge_metrics.i₊½.η̂.x₂[ᵢ₋₁]

  Jξx_ⱼ₊½ = edge_metrics.j₊½.ξ̂.x₁[idx]
  Jξy_ⱼ₊½ = edge_metrics.j₊½.ξ̂.x₂[idx]
  Jηx_ⱼ₊½ = edge_metrics.j₊½.η̂.x₁[idx]
  Jηy_ⱼ₊½ = edge_metrics.j₊½.η̂.x₂[idx]

  Jξx_ⱼ₋½ = edge_metrics.j₊½.ξ̂.x₁[ⱼ₋₁]
  Jξy_ⱼ₋½ = edge_metrics.j₊½.ξ̂.x₂[ⱼ₋₁]
  Jηx_ⱼ₋½ = edge_metrics.j₊½.η̂.x₁[ⱼ₋₁]
  Jηy_ⱼ₋½ = edge_metrics.j₊½.η̂.x₂[ⱼ₋₁]

  #------------------------------------------------------------------------------
  # Equations 3.43 and 3.44
  #------------------------------------------------------------------------------
  a_Jξ²ᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
  a_Jξ²ᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2) / Jᵢ₋½
  a_Jη²ⱼ₊½ = αⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½
  a_Jη²ⱼ₋½ = αⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2) / Jⱼ₋½

  a_Jξηᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½ * Jηx_ᵢ₊½ + Jξy_ᵢ₊½ * Jηy_ᵢ₊½) / (4Jᵢ₊½)
  a_Jξηᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½ * Jηx_ᵢ₋½ + Jξy_ᵢ₋½ * Jηy_ᵢ₋½) / (4Jᵢ₋½)
  a_Jηξⱼ₊½ = αⱼ₊½ * (Jξx_ⱼ₊½ * Jηx_ⱼ₊½ + Jξy_ⱼ₊½ * Jηy_ⱼ₊½) / (4Jⱼ₊½)
  a_Jηξⱼ₋½ = αⱼ₋½ * (Jξx_ⱼ₋½ * Jηx_ⱼ₋½ + Jξy_ⱼ₋½ * Jηy_ⱼ₋½) / (4Jⱼ₋½)

  return (; a_Jξ²ᵢ₊½, a_Jξ²ᵢ₋½, a_Jη²ⱼ₊½, a_Jη²ⱼ₋½, a_Jξηᵢ₊½, a_Jξηᵢ₋½, a_Jηξⱼ₊½, a_Jηξⱼ₋½)
end

@inline function non_conservative_metrics(
  cell_center_metrics, edge_metrics, idx::CartesianIndex{2}
)
  idim, jdim = (1, 2)
  ᵢ₋₁ = shift(idx, idim, -1)
  ⱼ₋₁ = shift(idx, jdim, -1)

  Jᵢ₊½ = edge_metrics.i₊½.J[idx]
  Jⱼ₊½ = edge_metrics.j₊½.J[idx]
  Jᵢ₋½ = edge_metrics.i₊½.J[ᵢ₋₁]
  Jⱼ₋½ = edge_metrics.j₊½.J[ⱼ₋₁]

  return (
    ξx=cell_center_metrics.ξ.x₁[idx],
    ξy=cell_center_metrics.ξ.x₂[idx],
    ηx=cell_center_metrics.η.x₁[idx],
    ηy=cell_center_metrics.η.x₂[idx],
    ξxᵢ₊½=edge_metrics.i₊½.ξ̂.x₁[idx] / Jᵢ₊½,
    ξyᵢ₊½=edge_metrics.i₊½.ξ̂.x₂[idx] / Jᵢ₊½,
    ηxᵢ₊½=edge_metrics.i₊½.η̂.x₁[idx] / Jᵢ₊½,
    ηyᵢ₊½=edge_metrics.i₊½.η̂.x₂[idx] / Jᵢ₊½,
    ξxᵢ₋½=edge_metrics.i₊½.ξ̂.x₁[ᵢ₋₁] / Jᵢ₋½,
    ξyᵢ₋½=edge_metrics.i₊½.ξ̂.x₂[ᵢ₋₁] / Jᵢ₋½,
    ηxᵢ₋½=edge_metrics.i₊½.η̂.x₁[ᵢ₋₁] / Jᵢ₋½,
    ηyᵢ₋½=edge_metrics.i₊½.η̂.x₂[ᵢ₋₁] / Jᵢ₋½,
    ξxⱼ₊½=edge_metrics.j₊½.ξ̂.x₁[idx] / Jⱼ₊½,
    ξyⱼ₊½=edge_metrics.j₊½.ξ̂.x₂[idx] / Jⱼ₊½,
    ηxⱼ₊½=edge_metrics.j₊½.η̂.x₁[idx] / Jⱼ₊½,
    ηyⱼ₊½=edge_metrics.j₊½.η̂.x₂[idx] / Jⱼ₊½,
    ξxⱼ₋½=edge_metrics.j₊½.ξ̂.x₁[ⱼ₋₁] / Jⱼ₋½,
    ξyⱼ₋½=edge_metrics.j₊½.ξ̂.x₂[ⱼ₋₁] / Jⱼ₋½,
    ηxⱼ₋½=edge_metrics.j₊½.η̂.x₁[ⱼ₋₁] / Jⱼ₋½,
    ηyⱼ₋½=edge_metrics.j₊½.η̂.x₂[ⱼ₋₁] / Jⱼ₋½,
  )
end

"""
  conservative_edge_terms(edge_diffusivity::NTuple{6,T}, m) where {T}

Collect and find the 3D edge terms used for the conservative form of the diffusion equation.
These are essentially the edge diffusivity + grid metric terms
"""
@inline function conservative_edge_terms(
  edge_diffusivity, edge_metrics, idx::CartesianIndex{3}
)
  @unpack αᵢ₊½, αᵢ₋½, αⱼ₊½, αⱼ₋½, αₖ₊½, αₖ₋½ = edge_diffusivity

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

  Jξx_ᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[idx]
  Jξy_ᵢ₊½ = edge_metrics.i₊½.ξ̂.x₂[idx]
  Jξz_ᵢ₊½ = edge_metrics.i₊½.ξ̂.x₃[idx]
  Jξx_ᵢ₋½ = edge_metrics.i₊½.ξ̂.x₁[ᵢ₋₁]
  Jξy_ᵢ₋½ = edge_metrics.i₊½.ξ̂.x₂[ᵢ₋₁]
  Jξz_ᵢ₋½ = edge_metrics.i₊½.ξ̂.x₃[ᵢ₋₁]
  Jηx_ᵢ₊½ = edge_metrics.i₊½.η̂.x₁[idx]
  Jηy_ᵢ₊½ = edge_metrics.i₊½.η̂.x₂[idx]
  Jηz_ᵢ₊½ = edge_metrics.i₊½.η̂.x₃[idx]
  Jηx_ᵢ₋½ = edge_metrics.i₊½.η̂.x₁[ᵢ₋₁]
  Jηy_ᵢ₋½ = edge_metrics.i₊½.η̂.x₂[ᵢ₋₁]
  Jηz_ᵢ₋½ = edge_metrics.i₊½.η̂.x₃[ᵢ₋₁]
  Jζx_ᵢ₊½ = edge_metrics.i₊½.ζ̂.x₁[idx]
  Jζy_ᵢ₊½ = edge_metrics.i₊½.ζ̂.x₂[idx]
  Jζz_ᵢ₊½ = edge_metrics.i₊½.ζ̂.x₃[idx]
  Jζx_ᵢ₋½ = edge_metrics.i₊½.ζ̂.x₁[ᵢ₋₁]
  Jζy_ᵢ₋½ = edge_metrics.i₊½.ζ̂.x₂[ᵢ₋₁]
  Jζz_ᵢ₋½ = edge_metrics.i₊½.ζ̂.x₃[ᵢ₋₁]

  Jξx_ⱼ₊½ = edge_metrics.j₊½.ξ̂.x₁[idx]
  Jξy_ⱼ₊½ = edge_metrics.j₊½.ξ̂.x₂[idx]
  Jξz_ⱼ₊½ = edge_metrics.j₊½.ξ̂.x₃[idx]
  Jξx_ⱼ₋½ = edge_metrics.j₊½.ξ̂.x₁[ⱼ₋₁]
  Jξy_ⱼ₋½ = edge_metrics.j₊½.ξ̂.x₂[ⱼ₋₁]
  Jξz_ⱼ₋½ = edge_metrics.j₊½.ξ̂.x₃[ⱼ₋₁]
  Jηx_ⱼ₊½ = edge_metrics.j₊½.η̂.x₁[idx]
  Jηy_ⱼ₊½ = edge_metrics.j₊½.η̂.x₂[idx]
  Jηz_ⱼ₊½ = edge_metrics.j₊½.η̂.x₃[idx]
  Jηx_ⱼ₋½ = edge_metrics.j₊½.η̂.x₁[ⱼ₋₁]
  Jηy_ⱼ₋½ = edge_metrics.j₊½.η̂.x₂[ⱼ₋₁]
  Jηz_ⱼ₋½ = edge_metrics.j₊½.η̂.x₃[ⱼ₋₁]
  Jζx_ⱼ₊½ = edge_metrics.j₊½.ζ̂.x₁[idx]
  Jζy_ⱼ₊½ = edge_metrics.j₊½.ζ̂.x₂[idx]
  Jζz_ⱼ₊½ = edge_metrics.j₊½.ζ̂.x₃[idx]
  Jζx_ⱼ₋½ = edge_metrics.j₊½.ζ̂.x₁[ⱼ₋₁]
  Jζy_ⱼ₋½ = edge_metrics.j₊½.ζ̂.x₂[ⱼ₋₁]
  Jζz_ⱼ₋½ = edge_metrics.j₊½.ζ̂.x₃[ⱼ₋₁]

  Jξx_ₖ₊½ = edge_metrics.k₊½.ξ̂.x₁[idx]
  Jξy_ₖ₊½ = edge_metrics.k₊½.ξ̂.x₂[idx]
  Jξz_ₖ₊½ = edge_metrics.k₊½.ξ̂.x₃[idx]
  Jξx_ₖ₋½ = edge_metrics.k₊½.ξ̂.x₁[ₖ₋₁]
  Jξy_ₖ₋½ = edge_metrics.k₊½.ξ̂.x₂[ₖ₋₁]
  Jξz_ₖ₋½ = edge_metrics.k₊½.ξ̂.x₃[ₖ₋₁]
  Jηx_ₖ₊½ = edge_metrics.k₊½.η̂.x₁[idx]
  Jηy_ₖ₊½ = edge_metrics.k₊½.η̂.x₂[idx]
  Jηz_ₖ₊½ = edge_metrics.k₊½.η̂.x₃[idx]
  Jηx_ₖ₋½ = edge_metrics.k₊½.η̂.x₁[ₖ₋₁]
  Jηy_ₖ₋½ = edge_metrics.k₊½.η̂.x₂[ₖ₋₁]
  Jηz_ₖ₋½ = edge_metrics.k₊½.η̂.x₃[ₖ₋₁]
  Jζx_ₖ₊½ = edge_metrics.k₊½.ζ̂.x₁[idx]
  Jζy_ₖ₊½ = edge_metrics.k₊½.ζ̂.x₂[idx]
  Jζz_ₖ₊½ = edge_metrics.k₊½.ζ̂.x₃[idx]
  Jζx_ₖ₋½ = edge_metrics.k₊½.ζ̂.x₁[ₖ₋₁]
  Jζy_ₖ₋½ = edge_metrics.k₊½.ζ̂.x₂[ₖ₋₁]
  Jζz_ₖ₋½ = edge_metrics.k₊½.ζ̂.x₃[ₖ₋₁]

  a_Jξ²ᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2 + Jξz_ᵢ₊½^2) / Jᵢ₊½
  a_Jξ²ᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2 + Jξz_ᵢ₋½^2) / Jᵢ₋½
  a_Jη²ⱼ₊½ = αⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2 + Jηz_ⱼ₊½^2) / Jⱼ₊½
  a_Jη²ⱼ₋½ = αⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2 + Jηz_ⱼ₋½^2) / Jⱼ₋½
  a_Jζ²ₖ₊½ = αₖ₊½ * (Jζx_ₖ₊½^2 + Jζy_ₖ₊½^2 + Jζz_ₖ₊½^2) / Jₖ₊½
  a_Jζ²ₖ₋½ = αₖ₋½ * (Jζx_ₖ₋½^2 + Jζy_ₖ₋½^2 + Jζz_ₖ₋½^2) / Jₖ₋½

  a_Jξηᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½ * Jηx_ᵢ₊½ + Jξy_ᵢ₊½ * Jηy_ᵢ₊½ + Jξz_ᵢ₊½ * Jηz_ᵢ₊½) / (4Jᵢ₊½)
  a_Jξηᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½ * Jηx_ᵢ₋½ + Jξy_ᵢ₋½ * Jηy_ᵢ₋½ + Jξz_ᵢ₋½ * Jηz_ᵢ₋½) / (4Jᵢ₋½)
  a_Jξζᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½ * Jζx_ᵢ₊½ + Jξy_ᵢ₊½ * Jζy_ᵢ₊½ + Jξz_ᵢ₊½ * Jζz_ᵢ₊½) / (4Jᵢ₊½)
  a_Jξζᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½ * Jζx_ᵢ₋½ + Jξy_ᵢ₋½ * Jζy_ᵢ₋½ + Jξz_ᵢ₋½ * Jζz_ᵢ₋½) / (4Jᵢ₋½)

  a_Jηξⱼ₊½ = αⱼ₊½ * (Jξx_ⱼ₊½ * Jηx_ⱼ₊½ + Jξy_ⱼ₊½ * Jηy_ⱼ₊½ + Jξz_ⱼ₊½ * Jηz_ⱼ₊½) / (4Jⱼ₊½)
  a_Jηξⱼ₋½ = αⱼ₋½ * (Jξx_ⱼ₋½ * Jηx_ⱼ₋½ + Jξy_ⱼ₋½ * Jηy_ⱼ₋½ + Jξz_ⱼ₋½ * Jηz_ⱼ₋½) / (4Jⱼ₋½)
  a_Jηζⱼ₊½ = αⱼ₊½ * (Jηx_ⱼ₊½ * Jζx_ⱼ₊½ + Jηy_ⱼ₊½ * Jζy_ⱼ₊½ + Jηz_ⱼ₊½ * Jζz_ⱼ₊½) / (4Jⱼ₊½)
  a_Jηζⱼ₋½ = αⱼ₋½ * (Jηx_ⱼ₋½ * Jζx_ⱼ₋½ + Jηy_ⱼ₋½ * Jζy_ⱼ₋½ + Jηz_ⱼ₋½ * Jζz_ⱼ₋½) / (4Jⱼ₋½)

  a_Jζξₖ₊½ = αₖ₊½ * (Jζx_ₖ₊½ * Jξx_ₖ₊½ + Jζy_ₖ₊½ * Jξy_ₖ₊½ + Jζz_ₖ₊½ * Jξz_ₖ₊½) / (4Jₖ₊½)
  a_Jζξₖ₋½ = αₖ₋½ * (Jζx_ₖ₋½ * Jξx_ₖ₋½ + Jζy_ₖ₋½ * Jξy_ₖ₋½ + Jζz_ₖ₋½ * Jξz_ₖ₋½) / (4Jₖ₋½)
  a_Jζηₖ₊½ = αₖ₊½ * (Jζx_ₖ₊½ * Jηx_ₖ₊½ + Jζy_ₖ₊½ * Jηy_ₖ₊½ + Jζz_ₖ₊½ * Jηz_ₖ₊½) / (4Jₖ₊½)
  a_Jζηₖ₋½ = αₖ₋½ * (Jζx_ₖ₋½ * Jηx_ₖ₋½ + Jζy_ₖ₋½ * Jηy_ₖ₋½ + Jζz_ₖ₋½ * Jηz_ₖ₋½) / (4Jₖ₋½)

  return (;
    a_Jξ²ᵢ₊½,
    a_Jξ²ᵢ₋½,
    a_Jξηᵢ₊½,
    a_Jξηᵢ₋½,
    a_Jξζᵢ₊½,
    a_Jξζᵢ₋½,
    a_Jηξⱼ₊½,
    a_Jηξⱼ₋½,
    a_Jη²ⱼ₊½,
    a_Jη²ⱼ₋½,
    a_Jηζⱼ₊½,
    a_Jηζⱼ₋½,
    a_Jζξₖ₊½,
    a_Jζξₖ₋½,
    a_Jζηₖ₊½,
    a_Jζηₖ₋½,
    a_Jζ²ₖ₊½,
    a_Jζ²ₖ₋½,
  )
end
