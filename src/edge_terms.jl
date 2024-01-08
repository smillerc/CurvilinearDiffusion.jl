
using UnPack

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

  fᵢ₊½ = αᵢ₊½ * (m.Jξx_ᵢ₊½^2) / m.Jᵢ₊½
  fᵢ₋½ = αᵢ₋½ * (m.Jξx_ᵢ₋½^2) / m.Jᵢ₋½

  return (; fᵢ₊½, fᵢ₋½)
end
"""
  conservative_edge_terms(edge_diffusivity::NTuple{4,T}, m) where {T}

Collect and find the 2D edge terms used for the conservative form of the diffusion equation.
These are essentially the edge diffusivity + grid metric terms
"""
@inline function conservative_edge_terms(edge_diffusivity::ET2D, m::NamedTuple)

  # m is a NamedTuple that contains the conservative edge metris for 
  # a single cell. The names should be self-explainatory
  @unpack αᵢ₊½, αᵢ₋½, αⱼ₊½, αⱼ₋½ = edge_diffusivity

  fᵢ₊½ = αᵢ₊½ * (m.Jξx_ᵢ₊½^2 + m.Jξy_ᵢ₊½^2) / m.Jᵢ₊½
  fᵢ₋½ = αᵢ₋½ * (m.Jξx_ᵢ₋½^2 + m.Jξy_ᵢ₋½^2) / m.Jᵢ₋½
  fⱼ₊½ = αⱼ₊½ * (m.Jηx_ⱼ₊½^2 + m.Jηy_ⱼ₊½^2) / m.Jⱼ₊½
  fⱼ₋½ = αⱼ₋½ * (m.Jηx_ⱼ₋½^2 + m.Jηy_ⱼ₋½^2) / m.Jⱼ₋½
  gᵢ₊½ = αᵢ₊½ * (m.Jξx_ᵢ₊½ * m.Jηx_ᵢ₊½ + m.Jξy_ᵢ₊½ * m.Jηy_ᵢ₊½) / (4m.Jᵢ₊½)
  gᵢ₋½ = αᵢ₋½ * (m.Jξx_ᵢ₋½ * m.Jηx_ᵢ₋½ + m.Jξy_ᵢ₋½ * m.Jηy_ᵢ₋½) / (4m.Jᵢ₋½)
  gⱼ₊½ = αⱼ₊½ * (m.Jξx_ⱼ₊½ * m.Jηx_ⱼ₊½ + m.Jξy_ⱼ₊½ * m.Jηy_ⱼ₊½) / (4m.Jⱼ₊½)
  gⱼ₋½ = αⱼ₋½ * (m.Jξx_ⱼ₋½ * m.Jηx_ⱼ₋½ + m.Jξy_ⱼ₋½ * m.Jηy_ⱼ₋½) / (4m.Jⱼ₋½)

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

  fᵢ₊½ = αᵢ₊½ * (m.Jξx_ᵢ₊½^2 + m.Jξy_ᵢ₊½^2 + m.Jξz_ᵢ₊½^2) / m.Jᵢ₊½
  fᵢ₋½ = αᵢ₋½ * (m.Jξx_ᵢ₋½^2 + m.Jξy_ᵢ₋½^2 + m.Jξz_ᵢ₋½^2) / m.Jᵢ₋½

  fⱼ₊½ = αⱼ₊½ * (m.Jηx_ⱼ₊½^2 + m.Jηy_ⱼ₊½^2 + m.Jηz_ⱼ₊½^2) / m.Jⱼ₊½
  fⱼ₋½ = αⱼ₋½ * (m.Jηx_ⱼ₋½^2 + m.Jηy_ⱼ₋½^2 + m.Jηz_ⱼ₋½^2) / m.Jⱼ₋½

  fₖ₊½ = αₖ₊½ * (m.Jζx_ₖ₊½^2 + m.Jζy_ₖ₊½^2 + m.Jζz_ₖ₊½^2) / m.Jₖ₊½
  fₖ₋½ = αₖ₋½ * (m.Jζx_ₖ₋½^2 + m.Jζy_ₖ₋½^2 + m.Jζz_ₖ₋½^2) / m.Jₖ₋½

  gᵢ₊½ =
    αᵢ₊½ * (
      m.Jξx_ᵢ₊½ * m.Jηx_ᵢ₊½ * m.Jζx_ᵢ₊½ + #
      m.Jξy_ᵢ₊½ * m.Jηy_ᵢ₊½ * m.Jζy_ᵢ₊½ + #
      m.Jξz_ᵢ₊½ * m.Jηz_ᵢ₊½ * m.Jζz_ᵢ₊½ #
    ) / (4m.Jᵢ₊½)

  gᵢ₋½ =
    αᵢ₋½ * (
      m.Jξx_ᵢ₋½ * m.Jηx_ᵢ₋½ * m.Jζx_ᵢ₋½ + #
      m.Jξy_ᵢ₋½ * m.Jηy_ᵢ₋½ * m.Jζy_ᵢ₋½ + #
      m.Jξz_ᵢ₋½ * m.Jηz_ᵢ₋½ * m.Jζz_ᵢ₋½ #
    ) / (4m.Jᵢ₋½)

  gⱼ₊½ =
    αⱼ₊½ * (
      m.Jξx_ⱼ₊½ * m.Jηx_ⱼ₊½ * m.Jζx_ⱼ₊½ + #
      m.Jξy_ⱼ₊½ * m.Jηy_ⱼ₊½ * m.Jζy_ⱼ₊½ + #
      m.Jξy_ⱼ₊½ * m.Jηy_ⱼ₊½ * m.Jζy_ⱼ₊½ #
    ) / (4m.Jⱼ₊½)

  gⱼ₋½ =
    αⱼ₋½ * (
      m.Jξx_ⱼ₋½ * m.Jηx_ⱼ₋½ * m.Jζx_ⱼ₋½ + #
      m.Jξy_ⱼ₋½ * m.Jηy_ⱼ₋½ * m.Jζy_ⱼ₋½ + #
      m.Jξz_ⱼ₋½ * m.Jηz_ⱼ₋½ * m.Jζz_ⱼ₋½ #
    ) / (4m.Jⱼ₋½)

  gₖ₊½ =
    αₖ₊½ * (
      m.Jξx_ₖ₊½ * m.Jηx_ₖ₊½ * m.Jζx_ₖ₊½ + #
      m.Jξy_ₖ₊½ * m.Jηy_ₖ₊½ * m.Jζy_ₖ₊½ + #
      m.Jξy_ₖ₊½ * m.Jηy_ₖ₊½ * m.Jζy_ₖ₊½ #
    ) / (4m.Jₖ₊½)

  gₖ₋½ =
    αₖ₋½ * (
      m.Jξx_ₖ₋½ * m.Jηx_ₖ₋½ * m.Jζx_ₖ₋½ + #
      m.Jξy_ₖ₋½ * m.Jηy_ₖ₋½ * m.Jζy_ₖ₋½ + #
      m.Jξz_ₖ₋½ * m.Jηz_ₖ₋½ * m.Jζz_ₖ₋½ #
    ) / (4m.Jₖ₋½)

  return (; fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, fₖ₊½, fₖ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½, gₖ₊½, gₖ₋½)
end
