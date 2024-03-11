
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

  Jξx_ᵢ₊½ = edge_metrics.i₊½.ξ̂.x[idx]
  Jξx_ᵢ₋½ = edge_metrics.i₊½.ξ̂.x[ᵢ₋₁]

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

  Jξx_ᵢ₊½ = edge_metrics.i₊½.ξ̂.x[idx]
  Jξy_ᵢ₊½ = edge_metrics.i₊½.ξ̂.y[idx]
  Jηx_ᵢ₊½ = edge_metrics.i₊½.η̂.x[idx]
  Jηy_ᵢ₊½ = edge_metrics.i₊½.η̂.y[idx]

  Jξx_ᵢ₋½ = edge_metrics.i₊½.ξ̂.x[ᵢ₋₁]
  Jξy_ᵢ₋½ = edge_metrics.i₊½.ξ̂.y[ᵢ₋₁]
  Jηx_ᵢ₋½ = edge_metrics.i₊½.η̂.x[ᵢ₋₁]
  Jηy_ᵢ₋½ = edge_metrics.i₊½.η̂.y[ᵢ₋₁]

  Jξx_ⱼ₊½ = edge_metrics.j₊½.ξ̂.x[idx]
  Jξy_ⱼ₊½ = edge_metrics.j₊½.ξ̂.y[idx]
  Jηx_ⱼ₊½ = edge_metrics.j₊½.η̂.x[idx]
  Jηy_ⱼ₊½ = edge_metrics.j₊½.η̂.y[idx]

  Jξx_ⱼ₋½ = edge_metrics.j₊½.ξ̂.x[ⱼ₋₁]
  Jξy_ⱼ₋½ = edge_metrics.j₊½.ξ̂.y[ⱼ₋₁]
  Jηx_ⱼ₋½ = edge_metrics.j₊½.η̂.x[ⱼ₋₁]
  Jηy_ⱼ₋½ = edge_metrics.j₊½.η̂.y[ⱼ₋₁]

  #------------------------------------------------------------------------------
  # Equations 3.43 and 3.44
  #------------------------------------------------------------------------------
  fᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
  fᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2) / Jᵢ₋½
  fⱼ₊½ = αⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½
  fⱼ₋½ = αⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2) / Jⱼ₋½
  # fᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
  # fᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2) / Jᵢ₋½
  # # i terms = fᵢ₋½ * u[i-1,j] - (fᵢ₊½ + fᵢ₋½) * u[i,j] + fᵢ₊½ * u[i+1,j]

  # fⱼ₊½ = a_ⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½
  # fⱼ₋½ = a_ⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2) / Jⱼ₋½
  # # j terms = fⱼ₋½ * u[i,j-1] - (fⱼ₊½ + fⱼ₋½) * u[i,j] + fⱼ₊½ * u[i,j+1]

  #------------------------------------------------------------------------------
  # cross terms (Equations 3.45 and 3.46)
  #------------------------------------------------------------------------------
  gᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½ * Jηx_ᵢ₊½ + Jξy_ᵢ₊½ * Jηy_ᵢ₊½) / (4Jᵢ₊½)
  gᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½ * Jηx_ᵢ₋½ + Jξy_ᵢ₋½ * Jηy_ᵢ₋½) / (4Jᵢ₋½)
  gⱼ₊½ = αⱼ₊½ * (Jξx_ⱼ₊½ * Jηx_ⱼ₊½ + Jξy_ⱼ₊½ * Jηy_ⱼ₊½) / (4Jⱼ₊½)
  gⱼ₋½ = αⱼ₋½ * (Jξx_ⱼ₋½ * Jηx_ⱼ₋½ + Jξy_ⱼ₋½ * Jηy_ⱼ₋½) / (4Jⱼ₋½)

  # gᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½ * Jηx_ᵢ₊½ + Jξy_ᵢ₊½ * Jηy_ᵢ₊½) / (4Jᵢ₊½)
  # gᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½ * Jηx_ᵢ₋½ + Jξy_ᵢ₋½ * Jηy_ᵢ₋½) / (4Jᵢ₋½)
  # # i terms = gᵢ₊½ * (u[i, j+1] − u[i, j-1] + u[i+1, j+1] − u[i+1, j-1])
  # #          -gᵢ₋½ * (u[i, j+1] − u[i, j-1] + u[j−1, j+1] − u[j−1, j-1])

  # gⱼ₊½ = a_ⱼ₊½ * (Jξx_ⱼ₊½ * Jηx_ⱼ₊½ + Jξy_ⱼ₊½ * Jηy_ⱼ₊½) / (4Jⱼ₊½)
  # gⱼ₋½ = a_ⱼ₋½ * (Jξx_ⱼ₋½ * Jηx_ⱼ₋½ + Jξy_ⱼ₋½ * Jηy_ⱼ₋½) / (4Jⱼ₋½)
  # # j terms = gⱼ₊½ * (u[i+1, j] − u[i-1, j] + u[i+1, j+1] − u[i-1, j+1])
  # #          -gⱼ₋½ * (u[i+1, j] − u[i-1, j] + u[i+1, j-1] − u[i-1, j-1])

  fᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
  fᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2) / Jᵢ₋½
  fⱼ₊½ = αⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½
  fⱼ₋½ = αⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2) / Jⱼ₋½
  gᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½ * Jηx_ᵢ₊½ + Jξy_ᵢ₊½ * Jηy_ᵢ₊½) / (4Jᵢ₊½)
  gᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½ * Jηx_ᵢ₋½ + Jξy_ᵢ₋½ * Jηy_ᵢ₋½) / (4Jᵢ₋½)
  gⱼ₊½ = αⱼ₊½ * (Jξx_ⱼ₊½ * Jηx_ⱼ₊½ + Jξy_ⱼ₊½ * Jηy_ⱼ₊½) / (4Jⱼ₊½)
  gⱼ₋½ = αⱼ₋½ * (Jξx_ⱼ₋½ * Jηx_ⱼ₋½ + Jξy_ⱼ₋½ * Jηy_ⱼ₋½) / (4Jⱼ₋½)

  return (; fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½)
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

  Jξx_ᵢ₊½ = edge_metrics.i₊½.ξ̂.x[idx]
  Jξy_ᵢ₊½ = edge_metrics.i₊½.ξ̂.y[idx]
  Jξz_ᵢ₊½ = edge_metrics.i₊½.ξ̂.z[idx]
  Jξx_ᵢ₋½ = edge_metrics.i₊½.ξ̂.x[ᵢ₋₁]
  Jξy_ᵢ₋½ = edge_metrics.i₊½.ξ̂.y[ᵢ₋₁]
  Jξz_ᵢ₋½ = edge_metrics.i₊½.ξ̂.z[ᵢ₋₁]
  Jηx_ᵢ₊½ = edge_metrics.i₊½.η̂.x[idx]
  Jηy_ᵢ₊½ = edge_metrics.i₊½.η̂.y[idx]
  Jηz_ᵢ₊½ = edge_metrics.i₊½.η̂.z[idx]
  Jηx_ᵢ₋½ = edge_metrics.i₊½.η̂.x[ᵢ₋₁]
  Jηy_ᵢ₋½ = edge_metrics.i₊½.η̂.y[ᵢ₋₁]
  Jηz_ᵢ₋½ = edge_metrics.i₊½.η̂.z[ᵢ₋₁]
  Jζx_ᵢ₊½ = edge_metrics.i₊½.ζ̂.x[idx]
  Jζy_ᵢ₊½ = edge_metrics.i₊½.ζ̂.y[idx]
  Jζz_ᵢ₊½ = edge_metrics.i₊½.ζ̂.z[idx]
  Jζx_ᵢ₋½ = edge_metrics.i₊½.ζ̂.x[ᵢ₋₁]
  Jζy_ᵢ₋½ = edge_metrics.i₊½.ζ̂.y[ᵢ₋₁]
  Jζz_ᵢ₋½ = edge_metrics.i₊½.ζ̂.z[ᵢ₋₁]

  Jξx_ⱼ₊½ = edge_metrics.j₊½.ξ̂.x[idx]
  Jξy_ⱼ₊½ = edge_metrics.j₊½.ξ̂.y[idx]
  Jξz_ⱼ₊½ = edge_metrics.j₊½.ξ̂.z[idx]
  Jξx_ⱼ₋½ = edge_metrics.j₊½.ξ̂.x[ⱼ₋₁]
  Jξy_ⱼ₋½ = edge_metrics.j₊½.ξ̂.y[ⱼ₋₁]
  Jξz_ⱼ₋½ = edge_metrics.j₊½.ξ̂.z[ⱼ₋₁]
  Jηx_ⱼ₊½ = edge_metrics.j₊½.η̂.x[idx]
  Jηy_ⱼ₊½ = edge_metrics.j₊½.η̂.y[idx]
  Jηz_ⱼ₊½ = edge_metrics.j₊½.η̂.z[idx]
  Jηx_ⱼ₋½ = edge_metrics.j₊½.η̂.x[ⱼ₋₁]
  Jηy_ⱼ₋½ = edge_metrics.j₊½.η̂.y[ⱼ₋₁]
  Jηz_ⱼ₋½ = edge_metrics.j₊½.η̂.z[ⱼ₋₁]
  Jζx_ⱼ₊½ = edge_metrics.j₊½.ζ̂.x[idx]
  Jζy_ⱼ₊½ = edge_metrics.j₊½.ζ̂.y[idx]
  Jζz_ⱼ₊½ = edge_metrics.j₊½.ζ̂.z[idx]
  Jζx_ⱼ₋½ = edge_metrics.j₊½.ζ̂.x[ⱼ₋₁]
  Jζy_ⱼ₋½ = edge_metrics.j₊½.ζ̂.y[ⱼ₋₁]
  Jζz_ⱼ₋½ = edge_metrics.j₊½.ζ̂.z[ⱼ₋₁]

  Jξx_ₖ₊½ = edge_metrics.k₊½.ξ̂.x[idx]
  Jξy_ₖ₊½ = edge_metrics.k₊½.ξ̂.y[idx]
  Jξz_ₖ₊½ = edge_metrics.k₊½.ξ̂.z[idx]
  Jξx_ₖ₋½ = edge_metrics.k₊½.ξ̂.x[ₖ₋₁]
  Jξy_ₖ₋½ = edge_metrics.k₊½.ξ̂.y[ₖ₋₁]
  Jξz_ₖ₋½ = edge_metrics.k₊½.ξ̂.z[ₖ₋₁]
  Jηx_ₖ₊½ = edge_metrics.k₊½.η̂.x[idx]
  Jηy_ₖ₊½ = edge_metrics.k₊½.η̂.y[idx]
  Jηz_ₖ₊½ = edge_metrics.k₊½.η̂.z[idx]
  Jηx_ₖ₋½ = edge_metrics.k₊½.η̂.x[ₖ₋₁]
  Jηy_ₖ₋½ = edge_metrics.k₊½.η̂.y[ₖ₋₁]
  Jηz_ₖ₋½ = edge_metrics.k₊½.η̂.z[ₖ₋₁]
  Jζx_ₖ₊½ = edge_metrics.k₊½.ζ̂.x[idx]
  Jζy_ₖ₊½ = edge_metrics.k₊½.ζ̂.y[idx]
  Jζz_ₖ₊½ = edge_metrics.k₊½.ζ̂.z[idx]
  Jζx_ₖ₋½ = edge_metrics.k₊½.ζ̂.x[ₖ₋₁]
  Jζy_ₖ₋½ = edge_metrics.k₊½.ζ̂.y[ₖ₋₁]
  Jζz_ₖ₋½ = edge_metrics.k₊½.ζ̂.z[ₖ₋₁]

  fᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2 + Jξz_ᵢ₊½^2) / Jᵢ₊½
  fᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2 + Jξz_ᵢ₋½^2) / Jᵢ₋½
  fⱼ₊½ = αⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2 + Jηz_ⱼ₊½^2) / Jⱼ₊½
  fⱼ₋½ = αⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2 + Jηz_ⱼ₋½^2) / Jⱼ₋½
  fₖ₊½ = αₖ₊½ * (Jζx_ₖ₊½^2 + Jζy_ₖ₊½^2 + Jζz_ₖ₊½^2) / Jₖ₊½
  fₖ₋½ = αₖ₋½ * (Jζx_ₖ₋½^2 + Jζy_ₖ₋½^2 + Jζz_ₖ₋½^2) / Jₖ₋½

  # TODO: these need to be verified
  gᵢ₊½ =
    αᵢ₊½ * (
      Jξx_ᵢ₊½ * Jηx_ᵢ₊½ * Jζx_ᵢ₊½ + #
      Jξy_ᵢ₊½ * Jηy_ᵢ₊½ * Jζy_ᵢ₊½ + #
      Jξz_ᵢ₊½ * Jηz_ᵢ₊½ * Jζz_ᵢ₊½ #
    ) / (4Jᵢ₊½)

  gᵢ₋½ =
    αᵢ₋½ * (
      Jξx_ᵢ₋½ * Jηx_ᵢ₋½ * Jζx_ᵢ₋½ + #
      Jξy_ᵢ₋½ * Jηy_ᵢ₋½ * Jζy_ᵢ₋½ + #
      Jξz_ᵢ₋½ * Jηz_ᵢ₋½ * Jζz_ᵢ₋½ #
    ) / (4Jᵢ₋½)

  gⱼ₊½ =
    αⱼ₊½ * (
      Jξx_ⱼ₊½ * Jηx_ⱼ₊½ * Jζx_ⱼ₊½ + #
      Jξy_ⱼ₊½ * Jηy_ⱼ₊½ * Jζy_ⱼ₊½ + #
      Jξz_ⱼ₊½ * Jηz_ⱼ₊½ * Jζz_ⱼ₊½ #
    ) / (4Jⱼ₊½)

  gⱼ₋½ =
    αⱼ₋½ * (
      Jξx_ⱼ₋½ * Jηx_ⱼ₋½ * Jζx_ⱼ₋½ + #
      Jξy_ⱼ₋½ * Jηy_ⱼ₋½ * Jζy_ⱼ₋½ + #
      Jξz_ⱼ₋½ * Jηz_ⱼ₋½ * Jζz_ⱼ₋½ #
    ) / (4Jⱼ₋½)

  gₖ₊½ =
    αₖ₊½ * (
      Jξx_ₖ₊½ * Jηx_ₖ₊½ * Jζx_ₖ₊½ + #
      Jξy_ₖ₊½ * Jηy_ₖ₊½ * Jζy_ₖ₊½ + #
      Jξz_ₖ₊½ * Jηz_ₖ₊½ * Jζz_ₖ₊½ #
    ) / (4Jₖ₊½)

  gₖ₋½ =
    αₖ₋½ * (
      Jξx_ₖ₋½ * Jηx_ₖ₋½ * Jζx_ₖ₋½ + #
      Jξy_ₖ₋½ * Jηy_ₖ₋½ * Jζy_ₖ₋½ + #
      Jξz_ₖ₋½ * Jηz_ₖ₋½ * Jζz_ₖ₋½ #
    ) / (4Jₖ₋½)

  return (; fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, fₖ₊½, fₖ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½, gₖ₊½, gₖ₋½)
end
