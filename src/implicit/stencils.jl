
function stencil_1d(edge_terms, J, Δt)
  @unpack fᵢ₊½, fᵢ₋½ = edge_terms

  uᵢ = -(fᵢ₋½ + fᵢ₊½) * Δt - J
  uᵢ₊₁ = fᵢ₊½ * Δt
  uᵢ₋₁ = fᵢ₋½ * Δt

  stencil = SVector(uᵢ₋₁, uᵢ, uᵢ₊₁)

  return stencil
end

# function stencil_2d(edge_terms, J, Δt)
#   @unpack fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½ = edge_terms
#   uᵢⱼ = -(fᵢ₋½ + fⱼ₋½ + fᵢ₊½ + fⱼ₊½) * Δt - J

#   # cardinal terms
#   uᵢ₊₁ⱼ = (fᵢ₊½ + gⱼ₊½ - gⱼ₋½) * Δt
#   uᵢ₋₁ⱼ = (fᵢ₋½ - gⱼ₊½ + gⱼ₋½) * Δt
#   uᵢⱼ₊₁ = (fⱼ₊½ + gᵢ₊½ - gᵢ₋½) * Δt
#   uᵢⱼ₋₁ = (fⱼ₋½ - gᵢ₊½ + gᵢ₋½) * Δt

#   # corner terms
#   uᵢ₊₁ⱼ₋₁ = (-gᵢ₊½ - gⱼ₋½) * Δt
#   uᵢ₊₁ⱼ₊₁ = (gᵢ₊½ + gⱼ₊½) * Δt
#   uᵢ₋₁ⱼ₋₁ = (gᵢ₋½ + gⱼ₋½) * Δt
#   uᵢ₋₁ⱼ₊₁ = (-gᵢ₋½ - gⱼ₊½) * Δt
#   stencil = SVector(uᵢ₋₁ⱼ₋₁, uᵢⱼ₋₁, uᵢ₊₁ⱼ₋₁, uᵢ₋₁ⱼ, uᵢⱼ, uᵢ₊₁ⱼ, uᵢ₋₁ⱼ₊₁, uᵢⱼ₊₁, uᵢ₊₁ⱼ₊₁)
#   return stencil
# end

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

# function neumann_stencil_2d(edge_terms, J, Δt, idx::CartesianIndex{2}, mesh_limits)
#   @unpack ilo, ihi, jlo, jhi = mesh_limits

#   T = eltype(edge_terms)

#   i, j = idx.I

#   a_Jξ²ᵢ₊½ = edge_terms.a_Jξ²ᵢ₊½
#   a_Jξ²ᵢ₋½ = edge_terms.a_Jξ²ᵢ₋½
#   a_Jη²ⱼ₊½ = edge_terms.a_Jη²ⱼ₊½
#   a_Jη²ⱼ₋½ = edge_terms.a_Jη²ⱼ₋½
#   a_Jξηᵢ₊½ = edge_terms.a_Jξηᵢ₊½
#   a_Jξηᵢ₋½ = edge_terms.a_Jξηᵢ₋½
#   a_Jηξⱼ₊½ = edge_terms.a_Jηξⱼ₊½
#   a_Jηξⱼ₋½ = edge_terms.a_Jηξⱼ₋½

#   if i == ilo
#     a_Jξ²ᵢ₋½ = a_Jξηᵢ₋½ = zero(T)
#   end

#   if j == jlo
#     a_Jη²ⱼ₋½ = a_Jηξⱼ₋½ = zero(T)
#   end

#   if i == ihi
#     a_Jξ²ᵢ₊½ = a_Jξηᵢ₊½ = zero(T)
#   end

#   if j == jhi
#     a_Jη²ⱼ₊½ = a_Jηξⱼ₊½ = zero(T)
#   end

#   uᵢⱼ = J + (a_Jη²ⱼ₊½ + a_Jη²ⱼ₋½ + a_Jξ²ᵢ₊½ + a_Jξ²ᵢ₋½) * Δt

#   uᵢ₊₁ⱼ₋₁ = (a_Jηξⱼ₋½ + a_Jξηᵢ₊½) * Δt
#   uᵢ₋₁ⱼ₊₁ = (a_Jηξⱼ₊½ + a_Jξηᵢ₋½) * Δt
#   uᵢ₋₁ⱼ₋₁ = (-a_Jηξⱼ₋½ - a_Jξηᵢ₋½) * Δt
#   uᵢ₊₁ⱼ₊₁ = (-a_Jηξⱼ₊½ - a_Jξηᵢ₊½) * Δt

#   uᵢ₊₁ⱼ = (-a_Jηξⱼ₊½ + a_Jηξⱼ₋½ - a_Jξ²ᵢ₊½) * Δt
#   uᵢ₋₁ⱼ = (a_Jηξⱼ₊½ - a_Jηξⱼ₋½ - a_Jξ²ᵢ₋½) * Δt
#   uᵢⱼ₊₁ = (-a_Jη²ⱼ₊½ - a_Jξηᵢ₊½ + a_Jξηᵢ₋½) * Δt
#   uᵢⱼ₋₁ = (-a_Jη²ⱼ₋½ + a_Jξηᵢ₊½ - a_Jξηᵢ₋½) * Δt

#   stencil = SVector(uᵢ₋₁ⱼ₋₁, uᵢⱼ₋₁, uᵢ₊₁ⱼ₋₁, uᵢ₋₁ⱼ, uᵢⱼ, uᵢ₊₁ⱼ, uᵢ₋₁ⱼ₊₁, uᵢⱼ₊₁, uᵢ₊₁ⱼ₊₁)
#   rhs_coeffs = @SVector zeros(T, 9) # nothing goes on the rhs for neumann bcs

#   return stencil, rhs_coeffs
# end

# function dirichlet_stencil_2d(edge_terms, J, Δt, idx::CartesianIndex{2}, mesh_limits)
#   # for the dirichlet bc, the boundary coefficients move to the right-hand-side, or b vector
#   # so we zero the coeffcients

#   @unpack ilo, ihi, jlo, jhi = mesh_limits

#   T = eltype(edge_terms)

#   i, j = idx.I

#   uᵢ₋₁ⱼ₋₁, uᵢⱼ₋₁, uᵢ₊₁ⱼ₋₁, uᵢ₋₁ⱼ, uᵢⱼ, uᵢ₊₁ⱼ, uᵢ₋₁ⱼ₊₁, uᵢⱼ₊₁, uᵢ₊₁ⱼ₊₁ = stencil_2d(
#     edge_terms, J, Δt
#   )

#   # keep all the coefficients for the rhs for now. the rhs update will use the proper terms
#   rhs_coeffs = SVector(uᵢ₋₁ⱼ₋₁, uᵢⱼ₋₁, uᵢ₊₁ⱼ₋₁, uᵢ₋₁ⱼ, uᵢⱼ, uᵢ₊₁ⱼ, uᵢ₋₁ⱼ₊₁, uᵢⱼ₊₁, uᵢ₊₁ⱼ₊₁)

#   # For the A matrix coefficients, zero out the terms that correspond to the boundary
#   if i == ilo && j == jlo
#     uᵢ₋₁ⱼ₋₁ = zero(T)
#   elseif i == ilo && j == jhi
#     uᵢ₋₁ⱼ₊₁ = zero(T)
#   elseif i == ihi && j == jlo
#     uᵢ₊₁ⱼ₋₁ = zero(T)
#   elseif i == ihi && j == jhi
#     uᵢ₊₁ⱼ₊₁ = zero(T)
#   elseif i == ilo
#     uᵢ₋₁ⱼ = zero(T)
#   elseif i == ihi
#     uᵢ₊₁ⱼ = zero(T)
#   elseif j == jlo
#     uᵢⱼ₋₁ = zero(T)
#   elseif j == jhi
#     uᵢⱼ₊₁ = zero(T)
#   end

#   stencil = SVector(uᵢ₋₁ⱼ₋₁, uᵢⱼ₋₁, uᵢ₊₁ⱼ₋₁, uᵢ₋₁ⱼ, uᵢⱼ, uᵢ₊₁ⱼ, uᵢ₋₁ⱼ₊₁, uᵢⱼ₊₁, uᵢ₊₁ⱼ₊₁)

#   return stencil, rhs_coeffs
# end

# function stencil_3d(edge_terms, J::T, Δt) where {T}
#   a_Jξ²ᵢ₊½ = edge_terms.a_Jξ²ᵢ₊½
#   a_Jξ²ᵢ₋½ = edge_terms.a_Jξ²ᵢ₋½
#   a_Jξηᵢ₊½ = edge_terms.a_Jξηᵢ₊½
#   a_Jξηᵢ₋½ = edge_terms.a_Jξηᵢ₋½
#   a_Jξζᵢ₊½ = edge_terms.a_Jξζᵢ₊½
#   a_Jξζᵢ₋½ = edge_terms.a_Jξζᵢ₋½
#   a_Jηξⱼ₊½ = edge_terms.a_Jηξⱼ₊½
#   a_Jηξⱼ₋½ = edge_terms.a_Jηξⱼ₋½
#   a_Jη²ⱼ₊½ = edge_terms.a_Jη²ⱼ₊½
#   a_Jη²ⱼ₋½ = edge_terms.a_Jη²ⱼ₋½
#   a_Jηζⱼ₊½ = edge_terms.a_Jηζⱼ₊½
#   a_Jηζⱼ₋½ = edge_terms.a_Jηζⱼ₋½
#   a_Jζξₖ₊½ = edge_terms.a_Jζξₖ₊½
#   a_Jζξₖ₋½ = edge_terms.a_Jζξₖ₋½
#   a_Jζηₖ₊½ = edge_terms.a_Jζηₖ₊½
#   a_Jζηₖ₋½ = edge_terms.a_Jζηₖ₋½
#   a_Jζ²ₖ₊½ = edge_terms.a_Jζ²ₖ₊½
#   a_Jζ²ₖ₋½ = edge_terms.a_Jζ²ₖ₋½

#   uᵢ₋₁ⱼ₋₁ₖ₋₁ = zero(T)
#   uᵢ₊₁ⱼ₋₁ₖ₋₁ = zero(T)
#   uᵢ₋₁ⱼ₊₁ₖ₋₁ = zero(T)
#   uᵢ₊₁ⱼ₊₁ₖ₋₁ = zero(T)
#   uᵢ₋₁ⱼ₋₁ₖ₊₁ = zero(T)
#   uᵢ₊₁ⱼ₋₁ₖ₊₁ = zero(T)
#   uᵢ₊₁ⱼ₊₁ₖ₊₁ = zero(T)
#   uᵢ₋₁ⱼ₊₁ₖ₊₁ = zero(T)

#   uᵢⱼ₋₁ₖ₋₁ = (-a_Jζηₖ₋½ - a_Jηζⱼ₋½) * Δt
#   uᵢ₋₁ⱼₖ₋₁ = (-a_Jζξₖ₋½ - a_Jξζᵢ₋½) * Δt
#   uᵢ₊₁ⱼₖ₋₁ = (a_Jζξₖ₋½ + a_Jξζᵢ₊½) * Δt
#   uᵢⱼ₊₁ₖ₋₁ = (a_Jζηₖ₋½ + a_Jηζⱼ₊½) * Δt
#   uᵢ₋₁ⱼ₋₁ₖ = (-a_Jηξⱼ₋½ - a_Jξηᵢ₋½) * Δt
#   uᵢ₊₁ⱼ₋₁ₖ = (a_Jηξⱼ₋½ + a_Jξηᵢ₊½) * Δt
#   uᵢ₋₁ⱼ₊₁ₖ = (a_Jηξⱼ₊½ + a_Jξηᵢ₋½) * Δt
#   uᵢ₊₁ⱼ₊₁ₖ = (-a_Jηξⱼ₊½ - a_Jξηᵢ₊½) * Δt
#   uᵢⱼ₋₁ₖ₊₁ = (a_Jζηₖ₊½ + a_Jηζⱼ₋½) * Δt
#   uᵢ₋₁ⱼₖ₊₁ = (a_Jζξₖ₊½ + a_Jξζᵢ₋½) * Δt
#   uᵢ₊₁ⱼₖ₊₁ = (-a_Jζξₖ₊½ - a_Jξζᵢ₊½) * Δt
#   uᵢⱼ₊₁ₖ₊₁ = (-a_Jζηₖ₊½ - a_Jηζⱼ₊½) * Δt

#   uᵢⱼₖ = J + (a_Jζ²ₖ₊½ + a_Jζ²ₖ₋½ + a_Jη²ⱼ₊½ + a_Jη²ⱼ₋½ + a_Jξ²ᵢ₊½ + a_Jξ²ᵢ₋½) * Δt

#   uᵢ₋₁ⱼₖ = (+a_Jζξₖ₊½ - a_Jζξₖ₋½ + a_Jηξⱼ₊½ - a_Jηξⱼ₋½ - a_Jξ²ᵢ₋½) * Δt
#   uᵢ₊₁ⱼₖ = (-a_Jζξₖ₊½ + a_Jζξₖ₋½ - a_Jηξⱼ₊½ + a_Jηξⱼ₋½ - a_Jξ²ᵢ₊½) * Δt

#   uᵢⱼ₊₁ₖ = (-a_Jζηₖ₊½ + a_Jζηₖ₋½ - a_Jη²ⱼ₊½ - a_Jξηᵢ₊½ + a_Jξηᵢ₋½) * Δt
#   uᵢⱼₖ₊₁ = (-a_Jζ²ₖ₊½ - a_Jηζⱼ₊½ + a_Jηζⱼ₋½ - a_Jξζᵢ₊½ + a_Jξζᵢ₋½) * Δt

#   uᵢⱼₖ₋₁ = (-a_Jζ²ₖ₋½ + a_Jηζⱼ₊½ - a_Jηζⱼ₋½ + a_Jξζᵢ₊½ - a_Jξζᵢ₋½) * Δt
#   uᵢⱼ₋₁ₖ = (+a_Jζηₖ₊½ - a_Jζηₖ₋½ - a_Jη²ⱼ₋½ + a_Jξηᵢ₊½ - a_Jξηᵢ₋½) * Δt

#   stencil = SVector(
#     uᵢ₋₁ⱼ₋₁ₖ₋₁,
#     uᵢⱼ₋₁ₖ₋₁,
#     uᵢ₊₁ⱼ₋₁ₖ₋₁,
#     uᵢ₋₁ⱼₖ₋₁,
#     uᵢⱼₖ₋₁,
#     uᵢ₊₁ⱼₖ₋₁,
#     uᵢ₋₁ⱼ₊₁ₖ₋₁,
#     uᵢⱼ₊₁ₖ₋₁,
#     uᵢ₊₁ⱼ₊₁ₖ₋₁,
#     uᵢ₋₁ⱼ₋₁ₖ,
#     uᵢⱼ₋₁ₖ,
#     uᵢ₊₁ⱼ₋₁ₖ,
#     uᵢ₋₁ⱼₖ,
#     uᵢⱼₖ,
#     uᵢ₊₁ⱼₖ,
#     uᵢ₋₁ⱼ₊₁ₖ,
#     uᵢⱼ₊₁ₖ,
#     uᵢ₊₁ⱼ₊₁ₖ,
#     uᵢ₋₁ⱼ₋₁ₖ₊₁,
#     uᵢⱼ₋₁ₖ₊₁,
#     uᵢ₊₁ⱼ₋₁ₖ₊₁,
#     uᵢ₋₁ⱼₖ₊₁,
#     uᵢⱼₖ₊₁,
#     uᵢ₊₁ⱼₖ₊₁,
#     uᵢ₋₁ⱼ₊₁ₖ₊₁,
#     uᵢⱼ₊₁ₖ₊₁,
#     uᵢ₊₁ⱼ₊₁ₖ₊₁,
#   )
#   return stencil
# end

# function bc_stencil_2d(edge_terms, J, Δt, idx::CartesianIndex{2}, mesh_limits, bcs)
#   @unpack ilo, ihi, jlo, jhi = mesh_limits

#   ilo_bc = bcs.ilo
#   ihi_bc = bcs.ihi
#   jlo_bc = bcs.jlo
#   jhi_bc = bcs.jhi

#   T = eltype(edge_terms)

#   i, j = idx.I

#   a_Jξ²ᵢ₊½ = edge_terms.a_Jξ²ᵢ₊½
#   a_Jξ²ᵢ₋½ = edge_terms.a_Jξ²ᵢ₋½
#   a_Jη²ⱼ₊½ = edge_terms.a_Jη²ⱼ₊½
#   a_Jη²ⱼ₋½ = edge_terms.a_Jη²ⱼ₋½
#   a_Jξηᵢ₊½ = edge_terms.a_Jξηᵢ₊½
#   a_Jξηᵢ₋½ = edge_terms.a_Jξηᵢ₋½
#   a_Jηξⱼ₊½ = edge_terms.a_Jηξⱼ₊½
#   a_Jηξⱼ₋½ = edge_terms.a_Jηξⱼ₋½

#   rhs_coeff = zero(T)
#   if i == ilo
#     a_Jξ²ᵢ₋½ = a_Jξηᵢ₋½ = zero(T)
#   end

#   if j == jlo && jlo_bc isa NeumannBC
#     a_Jη²ⱼ₋½ = a_Jηξⱼ₋½ = zero(T)
#   end

#   if i == ihi && ihi_bc isa NeumannBC
#     a_Jξ²ᵢ₊½ = a_Jξηᵢ₊½ = zero(T)
#   end

#   if j == jhi && jhi_bc isa NeumannBC
#     a_Jη²ⱼ₊½ = a_Jηξⱼ₊½ = zero(T)
#   end

#   Aᵢⱼ = J + (a_Jη²ⱼ₊½ + a_Jη²ⱼ₋½ + a_Jξ²ᵢ₊½ + a_Jξ²ᵢ₋½) * Δt

#   Aᵢ₊₁ⱼ₋₁ = (a_Jηξⱼ₋½ + a_Jξηᵢ₊½) * Δt
#   Aᵢ₋₁ⱼ₊₁ = (a_Jηξⱼ₊½ + a_Jξηᵢ₋½) * Δt
#   Aᵢ₋₁ⱼ₋₁ = (-a_Jηξⱼ₋½ - a_Jξηᵢ₋½) * Δt
#   Aᵢ₊₁ⱼ₊₁ = (-a_Jηξⱼ₊½ - a_Jξηᵢ₊½) * Δt

#   Aᵢ₊₁ⱼ = (-a_Jηξⱼ₊½ + a_Jηξⱼ₋½ - a_Jξ²ᵢ₊½) * Δt
#   Aᵢ₋₁ⱼ = (a_Jηξⱼ₊½ - a_Jηξⱼ₋½ - a_Jξ²ᵢ₋½) * Δt
#   Aᵢⱼ₊₁ = (-a_Jη²ⱼ₊½ - a_Jξηᵢ₊½ + a_Jξηᵢ₋½) * Δt
#   Aᵢⱼ₋₁ = (-a_Jη²ⱼ₋½ + a_Jξηᵢ₊½ - a_Jξηᵢ₋½) * Δt

#   A_coeff = SVector(Aᵢ₋₁ⱼ₋₁, Aᵢⱼ₋₁, Aᵢ₊₁ⱼ₋₁, Aᵢ₋₁ⱼ, Aᵢⱼ, Aᵢ₊₁ⱼ, Aᵢ₋₁ⱼ₊₁, Aᵢⱼ₊₁, Aᵢ₊₁ⱼ₊₁)

#   return A_coeff, rhs_coeff
# end

# function bc_coefficients()

#   #
#   @unpack ilo, ihi, jlo, jhi = mesh_limits
#   T = eltype(edge_terms)

#   ilo_bc = bcs.ilo
#   ihi_bc = bcs.ihi
#   jlo_bc = bcs.jlo
#   jhi_bc = bcs.jhi

#   i, j = idx.I

#   if i == ilo && j == jlo
#     _edge_terms = apply_edge_coeff_bc(ilo_bc, ILO_BC_LOC, edge_terms, idx)
#     bc_edge_terms = apply_edge_coeff_bc(jlo_bc, JLO_BC_LOC, _edge_terms, idx)
#   elseif i == ilo && j == jhi
#     _edge_terms = apply_edge_coeff_bc(ilo_bc, ILO_BC_LOC, edge_terms, idx)
#     bc_edge_terms = apply_edge_coeff_bc(jhi_bc, JHI_BC_LOC, _edge_terms, idx)
#   elseif i == ihi && j == jlo
#     _edge_terms = apply_edge_coeff_bc(ihi_bc, IHI_BC_LOC, edge_terms, idx)
#     bc_edge_terms = apply_edge_coeff_bc(jlo_bc, JLO_BC_LOC, _edge_terms, idx)
#   elseif i == ihi && j == jhi
#     _edge_terms = apply_edge_coeff_bc(ihi_bc, IHI_BC_LOC, edge_terms, idx)
#     bc_edge_terms = apply_edge_coeff_bc(jhi_bc, JHI_BC_LOC, _edge_terms, idx)
#   elseif i == ilo
#     bc_edge_terms = apply_edge_coeff_bc(ilo_bc, ILO_BC_LOC, edge_terms, idx)
#   elseif i == ihi
#     bc_edge_terms = apply_edge_coeff_bc(ihi_bc, IHI_BC_LOC, edge_terms, idx)
#   elseif j == jlo
#     bc_edge_terms = apply_edge_coeff_bc(jlo_bc, JLO_BC_LOC, edge_terms, idx)
#   elseif j == jhi
#     bc_edge_terms = apply_edge_coeff_bc(jhi_bc, JHI_BC_LOC, edge_terms, idx)
#   end

#   a_Jξ²ᵢ₊½ = bc_edge_terms.a_Jξ²ᵢ₊½
#   a_Jξ²ᵢ₋½ = bc_edge_terms.a_Jξ²ᵢ₋½
#   a_Jη²ⱼ₊½ = bc_edge_terms.a_Jη²ⱼ₊½
#   a_Jη²ⱼ₋½ = bc_edge_terms.a_Jη²ⱼ₋½
#   a_Jξηᵢ₊½ = bc_edge_terms.a_Jξηᵢ₊½
#   a_Jξηᵢ₋½ = bc_edge_terms.a_Jξηᵢ₋½
#   a_Jηξⱼ₊½ = bc_edge_terms.a_Jηξⱼ₊½
#   a_Jηξⱼ₋½ = bc_edge_terms.a_Jηξⱼ₋½

#   Aᵢⱼ = J + (a_Jη²ⱼ₊½ + a_Jη²ⱼ₋½ + a_Jξ²ᵢ₊½ + a_Jξ²ᵢ₋½) * Δt

#   Aᵢ₊₁ⱼ₋₁ = (a_Jηξⱼ₋½ + a_Jξηᵢ₊½) * Δt
#   Aᵢ₋₁ⱼ₊₁ = (a_Jηξⱼ₊½ + a_Jξηᵢ₋½) * Δt
#   Aᵢ₋₁ⱼ₋₁ = (-a_Jηξⱼ₋½ - a_Jξηᵢ₋½) * Δt
#   Aᵢ₊₁ⱼ₊₁ = (-a_Jηξⱼ₊½ - a_Jξηᵢ₊½) * Δt

#   Aᵢ₊₁ⱼ = (-a_Jηξⱼ₊½ + a_Jηξⱼ₋½ - a_Jξ²ᵢ₊½) * Δt
#   Aᵢ₋₁ⱼ = (a_Jηξⱼ₊½ - a_Jηξⱼ₋½ - a_Jξ²ᵢ₋½) * Δt
#   Aᵢⱼ₊₁ = (-a_Jη²ⱼ₊½ - a_Jξηᵢ₊½ + a_Jξηᵢ₋½) * Δt
#   Aᵢⱼ₋₁ = (-a_Jη²ⱼ₋½ + a_Jξηᵢ₊½ - a_Jξηᵢ₋½) * Δt

#   A_coeff = SVector(Aᵢ₋₁ⱼ₋₁, Aᵢⱼ₋₁, Aᵢ₊₁ⱼ₋₁, Aᵢ₋₁ⱼ, Aᵢⱼ, Aᵢ₊₁ⱼ, Aᵢ₋₁ⱼ₊₁, Aᵢⱼ₊₁, Aᵢ₊₁ⱼ₊₁)

#   if i == ilo && j == jlo
#   elseif i == ilo && j == jhi
#   elseif i == ihi && j == jlo
#   elseif i == ihi && j == jhi
#   elseif i == ilo
#   elseif i == ihi
#   elseif j == jlo
#   elseif j == jhi
#   end

#   return A_coeff, rhs_coeff
# end

# # alter the coefficients for a Neumann bc (i.e., zero-flux, insulated, zero-gradient, etc...)
# function apply_edge_coeff_bc(::NeumannBC, loc, edge_terms, ::CartesianIndex{2})
#   a_Jξ²ᵢ₊½ = edge_terms.a_Jξ²ᵢ₊½
#   a_Jξ²ᵢ₋½ = edge_terms.a_Jξ²ᵢ₋½
#   a_Jη²ⱼ₊½ = edge_terms.a_Jη²ⱼ₊½
#   a_Jη²ⱼ₋½ = edge_terms.a_Jη²ⱼ₋½
#   a_Jξηᵢ₊½ = edge_terms.a_Jξηᵢ₊½
#   a_Jξηᵢ₋½ = edge_terms.a_Jξηᵢ₋½
#   a_Jηξⱼ₊½ = edge_terms.a_Jηξⱼ₊½
#   a_Jηξⱼ₋½ = edge_terms.a_Jηξⱼ₋½

#   if loc == ILO_BC_LOC
#     a_Jξ²ᵢ₋½ = a_Jξηᵢ₋½ = zero(T)
#   elseif loc == IHI_BC_LOC
#     a_Jξ²ᵢ₊½ = a_Jξηᵢ₊½ = zero(T)
#   elseif loc == JLO_BC_LOC
#     a_Jη²ⱼ₋½ = a_Jηξⱼ₋½ = zero(T)
#   elseif loc == JHI_BC_LOC
#     a_Jη²ⱼ₊½ = a_Jηξⱼ₊½ = zero(T)
#   end

#   return (; a_Jξ²ᵢ₊½, a_Jξ²ᵢ₋½, a_Jη²ⱼ₊½, a_Jη²ⱼ₋½, a_Jξηᵢ₊½, a_Jξηᵢ₋½, a_Jηξⱼ₊½, a_Jηξⱼ₋½)
# end

# function apply_edge_coeff_bc(::DirichletBC, loc, edge_terms, ::CartesianIndex{2})
#   return edge_terms
# end

# function apply_A_coeff_bc(::NeumannBC, loc, edge_terms, ::CartesianIndex{2})
#   return edge_terms, rhs_term
# end

# function apply_A_coeff_bc(::DirichletBC, A_coeff::SVector{9,T}) where {T}
#   Aᵢ₋₁ⱼ₋₁, Aᵢⱼ₋₁, Aᵢ₊₁ⱼ₋₁, Aᵢ₋₁ⱼ, Aᵢⱼ, Aᵢ₊₁ⱼ, Aᵢ₋₁ⱼ₊₁, Aᵢⱼ₊₁, Aᵢ₊₁ⱼ₊₁ = A_coeff

#   rhs_coeff = zero(T)

#   if loc == ILO_BC_LOC
#     rhs_coeff = -(Aᵢ₋₁ⱼ₋₁ + Aᵢ₋₁ⱼ + Aᵢ₋₁ⱼ₊₁)
#     Aᵢ₋₁ⱼ₋₁ = Aᵢ₋₁ⱼ = Aᵢ₋₁ⱼ₊₁ = zero(T)
#   elseif loc == IHI_BC_LOC
#     Aᵢ₋₁ⱼ₋₁, Aᵢⱼ₋₁, Aᵢ₊₁ⱼ₋₁, Aᵢ₋₁ⱼ, Aᵢⱼ, Aᵢ₊₁ⱼ, Aᵢ₋₁ⱼ₊₁, Aᵢⱼ₊₁, Aᵢ₊₁ⱼ₊₁
#   elseif loc == JLO_BC_LOC
#     Aᵢ₋₁ⱼ₋₁, Aᵢⱼ₋₁, Aᵢ₊₁ⱼ₋₁, Aᵢ₋₁ⱼ, Aᵢⱼ, Aᵢ₊₁ⱼ, Aᵢ₋₁ⱼ₊₁, Aᵢⱼ₊₁, Aᵢ₊₁ⱼ₊₁
#   elseif loc == JHI_BC_LOC
#     Aᵢ₋₁ⱼ₋₁, Aᵢⱼ₋₁, Aᵢ₊₁ⱼ₋₁, Aᵢ₋₁ⱼ, Aᵢⱼ, Aᵢ₊₁ⱼ, Aᵢ₋₁ⱼ₊₁, Aᵢⱼ₊₁, Aᵢ₊₁ⱼ₊₁
#   end

#   A_coeff_updated = SVector(
#     Aᵢ₋₁ⱼ₋₁, Aᵢⱼ₋₁, Aᵢ₊₁ⱼ₋₁, Aᵢ₋₁ⱼ, Aᵢⱼ, Aᵢ₊₁ⱼ, Aᵢ₋₁ⱼ₊₁, Aᵢⱼ₊₁, Aᵢ₊₁ⱼ₊₁
#   )

#   return A_coeff_updated, rhs_coeff
# end

# # neumann (zero grad) alters the A_coeff by zeroing out the diffusivity from the boundary edge
# # dirichlet (fixed) alters A_coeff and rhs_coeff
