
function stencil_1d(edge_terms, J, Δt)
  @unpack fᵢ₊½, fᵢ₋½ = edge_terms

  uᵢ = -(fᵢ₋½ + fᵢ₊½) * Δt - J
  uᵢ₊₁ = fᵢ₊½ * Δt
  uᵢ₋₁ = fᵢ₋½ * Δt

  stencil = @SVector [uᵢ₋₁, uᵢ, uᵢ₊₁]

  return stencil
end

function stencil_2d(edge_terms, J, Δt)
  @unpack fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½ = edge_terms

  uᵢⱼ = -(fᵢ₋½ + fⱼ₋½ + fᵢ₊½ + fⱼ₊½) * Δt - J

  # cardinal terms
  uᵢ₊₁ⱼ = (fᵢ₊½ + gⱼ₊½ - gⱼ₋½) * Δt
  uᵢ₋₁ⱼ = (fᵢ₋½ - gⱼ₊½ + gⱼ₋½) * Δt
  uᵢⱼ₊₁ = (fⱼ₊½ + gᵢ₊½ - gᵢ₋½) * Δt
  uᵢⱼ₋₁ = (fⱼ₋½ - gᵢ₊½ + gᵢ₋½) * Δt

  # corner terms
  uᵢ₊₁ⱼ₋₁ = (-gᵢ₊½ - gⱼ₋½) * Δt
  uᵢ₊₁ⱼ₊₁ = (gᵢ₊½ + gⱼ₊½) * Δt
  uᵢ₋₁ⱼ₋₁ = (gᵢ₋½ + gⱼ₋½) * Δt
  uᵢ₋₁ⱼ₊₁ = (-gᵢ₋½ - gⱼ₊½) * Δt

  stencil = @SVector [uᵢ₋₁ⱼ₋₁, uᵢⱼ₋₁, uᵢ₊₁ⱼ₋₁, uᵢ₋₁ⱼ, uᵢⱼ, uᵢ₊₁ⱼ, uᵢ₋₁ⱼ₊₁, uᵢⱼ₊₁, uᵢ₊₁ⱼ₊₁]

  return stencil
end

function stencil_3d(edge_terms, J::T, Δt) where {T}
  fᵢ₊½ = edge_terms.fᵢ₊½
  fᵢ₋½ = edge_terms.fᵢ₋½
  fⱼ₊½ = edge_terms.fⱼ₊½
  fⱼ₋½ = edge_terms.fⱼ₋½
  fₖ₊½ = edge_terms.fₖ₊½
  fₖ₋½ = edge_terms.fₖ₋½

  gᵢ₊½ = edge_terms.gᵢ₊½
  gᵢ₋½ = edge_terms.gᵢ₋½
  gⱼ₊½ = edge_terms.gⱼ₊½
  gⱼ₋½ = edge_terms.gⱼ₋½
  gₖ₊½ = edge_terms.gₖ₊½
  gₖ₋½ = edge_terms.gₖ₋½

  hᵢ₊½ = edge_terms.hᵢ₊½
  hᵢ₋½ = edge_terms.hᵢ₋½
  hⱼ₊½ = edge_terms.hⱼ₊½
  hⱼ₋½ = edge_terms.hⱼ₋½
  hₖ₊½ = edge_terms.hₖ₊½
  hₖ₋½ = edge_terms.hₖ₋½

  uᵢⱼₖ = -(fᵢ₋½ + fⱼ₋½ + fₖ₋½ + fᵢ₊½ + fⱼ₊½ + fₖ₊½) * Δt - J

  uᵢ₊₁ⱼₖ = zero(T) * Δt
  uᵢ₋₁ⱼₖ = zero(T) * Δt
  uᵢⱼ₊₁ₖ = zero(T) * Δt
  uᵢⱼ₋₁ₖ = zero(T) * Δt
  uᵢⱼₖ₋₁ = zero(T) * Δt
  uᵢⱼₖ₊₁ = zero(T) * Δt

  uᵢ₋₁ⱼ₊₁ₖ = zero(T)
  uᵢ₊₁ⱼ₋₁ₖ = zero(T)
  uᵢ₊₁ⱼ₊₁ₖ = zero(T)
  uᵢ₋₁ⱼ₋₁ₖ = zero(T)
  uᵢ₊₁ⱼₖ₋₁ = zero(T)
  uᵢ₋₁ⱼₖ₊₁ = zero(T)
  uᵢ₊₁ⱼₖ₊₁ = zero(T)
  uᵢ₋₁ⱼₖ₋₁ = zero(T)
  uᵢⱼ₊₁ₖ₋₁ = zero(T)
  uᵢⱼ₋₁ₖ₊₁ = zero(T)
  uᵢⱼ₊₁ₖ₊₁ = zero(T)
  uᵢⱼ₋₁ₖ₋₁ = zero(T)

  uᵢ₊₁ⱼ₊₁ₖ₋₁ = zero(T)
  uᵢ₊₁ⱼ₊₁ₖ₊₁ = zero(T)
  uᵢ₊₁ⱼ₋₁ₖ₋₁ = zero(T)
  uᵢ₊₁ⱼ₋₁ₖ₊₁ = zero(T)
  uᵢ₋₁ⱼ₊₁ₖ₋₁ = zero(T)
  uᵢ₋₁ⱼ₊₁ₖ₊₁ = zero(T)
  uᵢ₋₁ⱼ₋₁ₖ₋₁ = zero(T)
  uᵢ₋₁ⱼ₋₁ₖ₊₁ = zero(T)

  stencil = @SVector [
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
  ]

  return stencil
end