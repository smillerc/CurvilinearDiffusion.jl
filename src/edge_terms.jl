
using CartesianDomainUtils

ET1D = @NamedTuple{αᵢ₊½::T1, αᵢ₋½::T2} where {T1,T2}
ET2D = @NamedTuple{αᵢ₊½::T1, αᵢ₋½::T2, αⱼ₊½::T3, αⱼ₋½::T4} where {T1,T2,T3,T4}
ET3D = @NamedTuple{
  αᵢ₊½::T1, αᵢ₋½::T2, αⱼ₊½::T3, αⱼ₋½::T4, αₖ₊½::T5, αₖ₋½::T6
} where {T1,T2,T3,T4,T5,T6}

"""
  conservative_edge_terms(edge_diffusivity::NTuple{4,T}, m) where {T}

Collect and find the 2D edge terms used for the conservative form of the diffusion equation.
These are essentially the edge diffusivity + grid metric terms
"""
@inline function conservative_edge_terms(edge_diffusivity::ET1D, m::NamedTuple)

  # m is a NamedTuple that contains the conservative edge metris for
  # a single cell. The names should be self-explainatory
  @unpack αᵢ₊½, αᵢ₋½ = edge_diffusivity

  Jᵢ₊½ = m.i₊½.J
  Jᵢ₋½ = m.i₋½.J

  Jξx_ᵢ₊½ = m.i₊½.ξx * Jᵢ₊½
  Jξx_ᵢ₋½ = m.i₋½.ξx * Jᵢ₋½

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

  # m is a NamedTuple that contains the conservative edge metris for
  # a single cell. The names should be self-explainatory
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
@inline function conservative_edge_terms(edge_diffusivity::ET3D, m::NamedTuple)

  # m is a NamedTuple that contains the conservative edge metris for
  # a single cell. The names should be self-explainatory
  @unpack αᵢ₊½, αᵢ₋½, αⱼ₊½, αⱼ₋½, αₖ₊½, αₖ₋½ = edge_diffusivity

  Jᵢ₊½ = m.i₊½.J
  Jᵢ₋½ = m.i₋½.J
  Jⱼ₋½ = m.j₋½.J
  Jⱼ₊½ = m.j₊½.J
  Jₖ₋½ = m.k₋½.J
  Jₖ₊½ = m.k₊½.J

  Jξx_ᵢ₊½ = m.i₊½.ξx * Jᵢ₊½
  Jξy_ᵢ₊½ = m.i₊½.ξy * Jᵢ₊½
  Jξz_ᵢ₊½ = m.i₊½.ξz * Jᵢ₊½
  Jξx_ᵢ₋½ = m.i₋½.ξx * Jᵢ₋½
  Jξy_ᵢ₋½ = m.i₋½.ξy * Jᵢ₋½
  Jξz_ᵢ₋½ = m.i₋½.ξz * Jᵢ₋½
  Jηx_ᵢ₊½ = m.i₊½.ηx * Jᵢ₊½
  Jηy_ᵢ₊½ = m.i₊½.ηy * Jᵢ₊½
  Jηz_ᵢ₊½ = m.i₊½.ηz * Jᵢ₊½
  Jηx_ᵢ₋½ = m.i₋½.ηx * Jᵢ₋½
  Jηy_ᵢ₋½ = m.i₋½.ηy * Jᵢ₋½
  Jηz_ᵢ₋½ = m.i₋½.ηz * Jᵢ₋½
  Jζx_ᵢ₊½ = m.i₊½.ζx * Jᵢ₊½
  Jζy_ᵢ₊½ = m.i₊½.ζy * Jᵢ₊½
  Jζz_ᵢ₊½ = m.i₊½.ζz * Jᵢ₊½
  Jζx_ᵢ₋½ = m.i₋½.ζx * Jᵢ₋½
  Jζy_ᵢ₋½ = m.i₋½.ζy * Jᵢ₋½
  Jζz_ᵢ₋½ = m.i₋½.ζz * Jᵢ₋½

  Jξx_ⱼ₊½ = m.j₊½.ξx * Jⱼ₊½
  Jξy_ⱼ₊½ = m.j₊½.ξy * Jⱼ₊½
  Jξz_ⱼ₊½ = m.j₊½.ξz * Jⱼ₊½
  Jξx_ⱼ₋½ = m.j₋½.ξx * Jⱼ₋½
  Jξy_ⱼ₋½ = m.j₋½.ξy * Jⱼ₋½
  Jξz_ⱼ₋½ = m.j₋½.ξz * Jⱼ₋½
  Jηx_ⱼ₊½ = m.j₊½.ηx * Jⱼ₊½
  Jηy_ⱼ₊½ = m.j₊½.ηy * Jⱼ₊½
  Jηz_ⱼ₊½ = m.j₊½.ηz * Jⱼ₊½
  Jηx_ⱼ₋½ = m.j₋½.ηx * Jⱼ₋½
  Jηy_ⱼ₋½ = m.j₋½.ηy * Jⱼ₋½
  Jηz_ⱼ₋½ = m.j₋½.ηz * Jⱼ₋½
  Jζx_ⱼ₊½ = m.j₊½.ζx * Jⱼ₊½
  Jζy_ⱼ₊½ = m.j₊½.ζy * Jⱼ₊½
  Jζz_ⱼ₊½ = m.j₊½.ζz * Jⱼ₊½
  Jζx_ⱼ₋½ = m.j₋½.ζx * Jⱼ₋½
  Jζy_ⱼ₋½ = m.j₋½.ζy * Jⱼ₋½
  Jζz_ⱼ₋½ = m.j₋½.ζz * Jⱼ₋½

  Jξx_ₖ₊½ = m.k₊½.ξx * Jₖ₊½
  Jξy_ₖ₊½ = m.k₊½.ξy * Jₖ₊½
  Jξz_ₖ₊½ = m.k₊½.ξz * Jₖ₊½
  Jξx_ₖ₋½ = m.k₋½.ξx * Jₖ₋½
  Jξy_ₖ₋½ = m.k₋½.ξy * Jₖ₋½
  Jξz_ₖ₋½ = m.k₋½.ξz * Jₖ₋½
  Jηx_ₖ₊½ = m.k₊½.ηx * Jₖ₊½
  Jηy_ₖ₊½ = m.k₊½.ηy * Jₖ₊½
  Jηz_ₖ₊½ = m.k₊½.ηz * Jₖ₊½
  Jηx_ₖ₋½ = m.k₋½.ηx * Jₖ₋½
  Jηy_ₖ₋½ = m.k₋½.ηy * Jₖ₋½
  Jηz_ₖ₋½ = m.k₋½.ηz * Jₖ₋½
  Jζx_ₖ₊½ = m.k₊½.ζx * Jₖ₊½
  Jζy_ₖ₊½ = m.k₊½.ζy * Jₖ₊½
  Jζz_ₖ₊½ = m.k₊½.ζz * Jₖ₊½
  Jζx_ₖ₋½ = m.k₋½.ζx * Jₖ₋½
  Jζy_ₖ₋½ = m.k₋½.ζy * Jₖ₋½
  Jζz_ₖ₋½ = m.k₋½.ζz * Jₖ₋½

  fᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2 + Jξz_ᵢ₊½^2) / Jᵢ₊½
  fᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2 + Jξz_ᵢ₋½^2) / Jᵢ₋½
  fⱼ₊½ = αⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2 + Jηz_ⱼ₊½^2) / Jⱼ₊½
  fⱼ₋½ = αⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2 + Jηz_ⱼ₋½^2) / Jⱼ₋½
  fₖ₊½ = αₖ₊½ * (Jζx_ₖ₊½^2 + Jζy_ₖ₊½^2 + Jζz_ₖ₊½^2) / Jₖ₊½
  fₖ₋½ = αₖ₋½ * (Jζx_ₖ₋½^2 + Jζy_ₖ₋½^2 + Jζz_ₖ₋½^2) / Jₖ₋½

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
