
function stencil_1d(edge_terms, J, Δt)
  @unpack fᵢ₊½, fᵢ₋½ = edge_terms

  uᵢ = -(fᵢ₋½ + fᵢ₊½) * Δt - J
  uᵢ₊₁ = fᵢ₊½ * Δt
  uᵢ₋₁ = fᵢ₋½ * Δt

  stencil = SVector(uᵢ₋₁, uᵢ, uᵢ₊₁)

  return stencil
end

function stencil_2d(edge_terms, J, Δt)
  a_Jξ²ᵢ₊½ = edge_terms.a_Jξ²ᵢ₊½
  a_Jξ²ᵢ₋½ = edge_terms.a_Jξ²ᵢ₋½
  a_Jη²ⱼ₊½ = edge_terms.a_Jη²ⱼ₊½
  a_Jη²ⱼ₋½ = edge_terms.a_Jη²ⱼ₋½
  a_Jξηᵢ₊½ = edge_terms.a_Jξηᵢ₊½
  a_Jξηᵢ₋½ = edge_terms.a_Jξηᵢ₋½
  a_Jηξⱼ₊½ = edge_terms.a_Jηξⱼ₊½
  a_Jηξⱼ₋½ = edge_terms.a_Jηξⱼ₋½

  uᵢⱼ = J + (a_Jη²ⱼ₊½ + a_Jη²ⱼ₋½ + a_Jξ²ᵢ₊½ + a_Jξ²ᵢ₋½) * Δt

  uᵢ₊₁ⱼ₋₁ = (a_Jηξⱼ₋½ + a_Jξηᵢ₊½) * Δt
  uᵢ₋₁ⱼ₊₁ = (a_Jηξⱼ₊½ + a_Jξηᵢ₋½) * Δt
  uᵢ₋₁ⱼ₋₁ = (-a_Jηξⱼ₋½ - a_Jξηᵢ₋½) * Δt
  uᵢ₊₁ⱼ₊₁ = (-a_Jηξⱼ₊½ - a_Jξηᵢ₊½) * Δt

  uᵢ₊₁ⱼ = (-a_Jηξⱼ₊½ + a_Jηξⱼ₋½ - a_Jξ²ᵢ₊½) * Δt
  uᵢ₋₁ⱼ = (a_Jηξⱼ₊½ - a_Jηξⱼ₋½ - a_Jξ²ᵢ₋½) * Δt
  uᵢⱼ₊₁ = (-a_Jη²ⱼ₊½ - a_Jξηᵢ₊½ + a_Jξηᵢ₋½) * Δt
  uᵢⱼ₋₁ = (-a_Jη²ⱼ₋½ + a_Jξηᵢ₊½ - a_Jξηᵢ₋½) * Δt
  stencil = SVector(uᵢ₋₁ⱼ₋₁, uᵢⱼ₋₁, uᵢ₊₁ⱼ₋₁, uᵢ₋₁ⱼ, uᵢⱼ, uᵢ₊₁ⱼ, uᵢ₋₁ⱼ₊₁, uᵢⱼ₊₁, uᵢ₊₁ⱼ₊₁)
  return stencil
end

function stencil_3d(edge_terms, J::T, Δt) where {T}
  a_Jξ²ᵢ₊½ = edge_terms.a_Jξ²ᵢ₊½
  a_Jξ²ᵢ₋½ = edge_terms.a_Jξ²ᵢ₋½
  a_Jξηᵢ₊½ = edge_terms.a_Jξηᵢ₊½
  a_Jξηᵢ₋½ = edge_terms.a_Jξηᵢ₋½
  a_Jξζᵢ₊½ = edge_terms.a_Jξζᵢ₊½
  a_Jξζᵢ₋½ = edge_terms.a_Jξζᵢ₋½
  a_Jηξⱼ₊½ = edge_terms.a_Jηξⱼ₊½
  a_Jηξⱼ₋½ = edge_terms.a_Jηξⱼ₋½
  a_Jη²ⱼ₊½ = edge_terms.a_Jη²ⱼ₊½
  a_Jη²ⱼ₋½ = edge_terms.a_Jη²ⱼ₋½
  a_Jηζⱼ₊½ = edge_terms.a_Jηζⱼ₊½
  a_Jηζⱼ₋½ = edge_terms.a_Jηζⱼ₋½
  a_Jζξₖ₊½ = edge_terms.a_Jζξₖ₊½
  a_Jζξₖ₋½ = edge_terms.a_Jζξₖ₋½
  a_Jζηₖ₊½ = edge_terms.a_Jζηₖ₊½
  a_Jζηₖ₋½ = edge_terms.a_Jζηₖ₋½
  a_Jζ²ₖ₊½ = edge_terms.a_Jζ²ₖ₊½
  a_Jζ²ₖ₋½ = edge_terms.a_Jζ²ₖ₋½

  uᵢ₋₁ⱼ₋₁ₖ₋₁ = zero(T)
  uᵢ₊₁ⱼ₋₁ₖ₋₁ = zero(T)
  uᵢ₋₁ⱼ₊₁ₖ₋₁ = zero(T)
  uᵢ₊₁ⱼ₊₁ₖ₋₁ = zero(T)
  uᵢ₋₁ⱼ₋₁ₖ₊₁ = zero(T)
  uᵢ₊₁ⱼ₋₁ₖ₊₁ = zero(T)
  uᵢ₊₁ⱼ₊₁ₖ₊₁ = zero(T)
  uᵢ₋₁ⱼ₊₁ₖ₊₁ = zero(T)

  uᵢⱼ₋₁ₖ₋₁ = (-a_Jζηₖ₋½ - a_Jηζⱼ₋½) * Δt
  uᵢ₋₁ⱼₖ₋₁ = (-a_Jζξₖ₋½ - a_Jξζᵢ₋½) * Δt
  uᵢ₊₁ⱼₖ₋₁ = (a_Jζξₖ₋½ + a_Jξζᵢ₊½) * Δt
  uᵢⱼ₊₁ₖ₋₁ = (a_Jζηₖ₋½ + a_Jηζⱼ₊½) * Δt
  uᵢ₋₁ⱼ₋₁ₖ = (-a_Jηξⱼ₋½ - a_Jξηᵢ₋½) * Δt
  uᵢ₊₁ⱼ₋₁ₖ = (a_Jηξⱼ₋½ + a_Jξηᵢ₊½) * Δt
  uᵢ₋₁ⱼ₊₁ₖ = (a_Jηξⱼ₊½ + a_Jξηᵢ₋½) * Δt
  uᵢ₊₁ⱼ₊₁ₖ = (-a_Jηξⱼ₊½ - a_Jξηᵢ₊½) * Δt
  uᵢⱼ₋₁ₖ₊₁ = (a_Jζηₖ₊½ + a_Jηζⱼ₋½) * Δt
  uᵢ₋₁ⱼₖ₊₁ = (a_Jζξₖ₊½ + a_Jξζᵢ₋½) * Δt
  uᵢ₊₁ⱼₖ₊₁ = (-a_Jζξₖ₊½ - a_Jξζᵢ₊½) * Δt
  uᵢⱼ₊₁ₖ₊₁ = (-a_Jζηₖ₊½ - a_Jηζⱼ₊½) * Δt

  uᵢⱼₖ = J + (a_Jζ²ₖ₊½ + a_Jζ²ₖ₋½ + a_Jη²ⱼ₊½ + a_Jη²ⱼ₋½ + a_Jξ²ᵢ₊½ + a_Jξ²ᵢ₋½) * Δt
  # @show edge_terms

  uᵢ₊₁ⱼₖ = (-a_Jξ²ᵢ₊½ - a_Jηξⱼ₊½ + a_Jηξⱼ₋½ - a_Jζξₖ₊½ + a_Jζξₖ₋½) * Δt
  uᵢ₋₁ⱼₖ = (-a_Jξ²ᵢ₋½ + a_Jηξⱼ₊½ - a_Jηξⱼ₋½ + a_Jζξₖ₊½ - a_Jζξₖ₋½) * Δt

  uᵢⱼ₊₁ₖ = (-a_Jη²ⱼ₊½ - a_Jξηᵢ₊½ + a_Jξηᵢ₋½ - a_Jζηₖ₊½ + a_Jζηₖ₋½) * Δt
  uᵢⱼ₋₁ₖ = (-a_Jη²ⱼ₋½ + a_Jξηᵢ₊½ - a_Jξηᵢ₋½ + a_Jζηₖ₊½ - a_Jζηₖ₋½) * Δt

  uᵢⱼₖ₊₁ = (-a_Jζ²ₖ₊½ - a_Jξζᵢ₊½ + a_Jξζᵢ₋½ - a_Jηζⱼ₊½ + a_Jηζⱼ₋½) * Δt
  uᵢⱼₖ₋₁ = (-a_Jζ²ₖ₋½ + a_Jξζᵢ₊½ - a_Jξζᵢ₋½ + a_Jηζⱼ₊½ - a_Jηζⱼ₋½) * Δt

  stencil = SVector(
    uᵢ₋₁ⱼ₋₁ₖ₋₁,
    uᵢⱼ₋₁ₖ₋₁,
    uᵢ₊₁ⱼ₋₁ₖ₋₁,
    uᵢ₋₁ⱼₖ₋₁,
    uᵢⱼₖ₋₁,
    uᵢ₊₁ⱼₖ₋₁,
    uᵢ₋₁ⱼ₊₁ₖ₋₁,
    uᵢⱼ₊₁ₖ₋₁,
    uᵢ₊₁ⱼ₊₁ₖ₋₁,
    uᵢ₋₁ⱼ₋₁ₖ,
    uᵢⱼ₋₁ₖ,
    uᵢ₊₁ⱼ₋₁ₖ,
    uᵢ₋₁ⱼₖ,
    uᵢⱼₖ,
    uᵢ₊₁ⱼₖ,
    uᵢ₋₁ⱼ₊₁ₖ,
    uᵢⱼ₊₁ₖ,
    uᵢ₊₁ⱼ₊₁ₖ,
    uᵢ₋₁ⱼ₋₁ₖ₊₁,
    uᵢⱼ₋₁ₖ₊₁,
    uᵢ₊₁ⱼ₋₁ₖ₊₁,
    uᵢ₋₁ⱼₖ₊₁,
    uᵢⱼₖ₊₁,
    uᵢ₊₁ⱼₖ₊₁,
    uᵢ₋₁ⱼ₊₁ₖ₊₁,
    uᵢⱼ₊₁ₖ₊₁,
    uᵢ₊₁ⱼ₊₁ₖ₊₁,
  )
  return stencil
end
