
function diffusion_op_2nd_order_2d_zerogradient_bc(
  edge_metrics, a_edge::SVector{4,T}, loc
) where {T}

  # Create a stencil matrix to hold the coefficients for u[i±1,j±1]

  # Don't use an offset matrix here b/c benchmarking showed it was
  # ~3x slower. It makes the indexing here more of a pain,
  # but 3x slower is a big deal for this performance critical section
  stencil = MMatrix{3,3,T}(0, 0, 0, 0, 0, 0, 0, 0, 0)

  # Here the Δξ and Δη terms are 1
  a_ᵢ₊½, a_ᵢ₋½, a_ⱼ₊½, a_ⱼ₋½ = a_edge

  # edge_metrics = metrics_2d(mesh, i, j)
  Jᵢ₊½ = edge_metrics.Jᵢ₊½
  Jᵢ₋½ = edge_metrics.Jᵢ₋½
  Jⱼ₊½ = edge_metrics.Jⱼ₊½
  Jⱼ₋½ = edge_metrics.Jⱼ₋½
  Jξx_ᵢ₊½ = edge_metrics.Jξx_ᵢ₊½
  Jξy_ᵢ₊½ = edge_metrics.Jξy_ᵢ₊½
  Jηx_ᵢ₊½ = edge_metrics.Jηx_ᵢ₊½
  Jηy_ᵢ₊½ = edge_metrics.Jηy_ᵢ₊½
  Jξx_ᵢ₋½ = edge_metrics.Jξx_ᵢ₋½
  Jξy_ᵢ₋½ = edge_metrics.Jξy_ᵢ₋½
  Jηx_ᵢ₋½ = edge_metrics.Jηx_ᵢ₋½
  Jηy_ᵢ₋½ = edge_metrics.Jηy_ᵢ₋½
  Jξx_ⱼ₊½ = edge_metrics.Jξx_ⱼ₊½
  Jξy_ⱼ₊½ = edge_metrics.Jξy_ⱼ₊½
  Jηx_ⱼ₊½ = edge_metrics.Jηx_ⱼ₊½
  Jηy_ⱼ₊½ = edge_metrics.Jηy_ⱼ₊½
  Jξx_ⱼ₋½ = edge_metrics.Jξx_ⱼ₋½
  Jξy_ⱼ₋½ = edge_metrics.Jξy_ⱼ₋½
  Jηx_ⱼ₋½ = edge_metrics.Jηx_ⱼ₋½
  Jηy_ⱼ₋½ = edge_metrics.Jηy_ⱼ₋½

  #------------------------------------------------------------------------------
  # Equations 3.43 and 3.44
  #------------------------------------------------------------------------------
  # Diff(I) term ∂/∂ξ [... ∂/∂ξ]
  fᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
  fᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2) / Jᵢ₋½
  # i terms = fᵢ₋½ * u[i-1,j] - (fᵢ₊½ + fᵢ₋½) * u[i,j] + fᵢ₊½ * u[i+1,j]

  # Diff(IV) term ∂/∂η [... ∂/∂η]
  fⱼ₊½ = a_ⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½
  fⱼ₋½ = a_ⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2) / Jⱼ₋½
  # j terms = fⱼ₋½ * u[i,j-1] - (fⱼ₊½ + fⱼ₋½) * u[i,j] + fⱼ₊½ * u[i,j+1]

  # i boundaries
  if loc === :ilo || loc === :ilojlo || loc === :ilojhi
    fᵢ₋½ = zero(T)
  elseif loc === :ihi || loc === :ihijlo || loc === :ihijhi
    fᵢ₊½ = zero(T)
  end

  stencil[1, 2] += fᵢ₋½           # u[i-1, j]
  stencil[2, 2] += -(fᵢ₊½ + fᵢ₋½) # u[i  , j]
  stencil[3, 2] += fᵢ₊½           # u[i+1, j]

  # j boundaries
  if loc === :jlo || loc === :ilojlo || loc === :ihijlo
    fⱼ₋½ = zero(T)
  elseif loc === :jhi || loc === :ilojhi || loc === :ihijhi
    fⱼ₋½ = zero(T)
  end

  stencil[2, 1] += fⱼ₋½           # u[i, j-1]
  stencil[2, 2] += -(fⱼ₊½ + fⱼ₋½) # u[i, j  ]
  stencil[2, 3] += fⱼ₊½           # u[i, j+1]

  #------------------------------------------------------------------------------
  # cross terms (Equations 3.45 and 3.46)
  #------------------------------------------------------------------------------
  # TODO: pure rectangular grids can skip this section

  # Diff(II) term  ∂/∂ξ [... ∂/∂η] -> only apply this on i boundaries,
  # since ∂/∂η = 0 at all the j boundaries
  if (loc === :ilo || loc === :ihi)
    gᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½ * Jηx_ᵢ₊½ + Jξy_ᵢ₊½ * Jηy_ᵢ₊½) / (4Jᵢ₊½)
    gᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½ * Jηx_ᵢ₋½ + Jξy_ᵢ₋½ * Jηy_ᵢ₋½) / (4Jᵢ₋½)
    # i terms = gᵢ₊½ * (u[i, j+1] − u[i, j-1] + u[i+1, j+1] − u[i+1, j-1])
    #          -gᵢ₋½ * (u[i, j+1] − u[i, j-1] + u[j−1, j+1] − u[j−1, j-1])

    stencil[1, 1] += gᵢ₋½           # u[i-1, j-1]
    stencil[2, 1] += (-gᵢ₊½ + gᵢ₋½) # u[i  , j-1]
    stencil[3, 1] += -gᵢ₊½          # u[i+1, j-1]
    stencil[1, 3] += -gᵢ₋½          # u[i-1, j+1]
    stencil[2, 3] += (gᵢ₊½ - gᵢ₋½)  # u[i  , j+1]
    stencil[3, 3] += gᵢ₊½           # u[i+1, j+1]
  end

  # Diff(III) term  ∂/∂η [... ∂/∂ξ] -> only apply this on j boundaries,
  # since ∂/∂ξ = 0 at all the i boundaries
  if (loc === :jlo || loc === :jhi)
    gⱼ₊½ = a_ⱼ₊½ * (Jξx_ⱼ₊½ * Jηx_ⱼ₊½ + Jξy_ⱼ₊½ * Jηy_ⱼ₊½) / (4Jⱼ₊½)
    gⱼ₋½ = a_ⱼ₋½ * (Jξx_ⱼ₋½ * Jηx_ⱼ₋½ + Jξy_ⱼ₋½ * Jηy_ⱼ₋½) / (4Jⱼ₋½)
    # j terms = gⱼ₊½ * (u[i+1, j] − u[i-1, j] + u[i+1, j+1] − u[i-1, j+1])
    #          -gⱼ₋½ * (u[i+1, j] − u[i-1, j] + u[i+1, j-1] − u[i-1, j-1])

    stencil[1, 1] += gⱼ₋½           # u[i-1, j-1]
    stencil[3, 1] += -gⱼ₋½          # u[i+1, j-1]
    stencil[1, 2] += (-gⱼ₊½ + gⱼ₋½) # u[i-1, j  ]
    stencil[3, 2] += (gⱼ₊½ - gⱼ₋½)  # u[i+1, j  ]
    stencil[1, 3] += -gⱼ₊½          # u[i-1, j+1]
    stencil[3, 3] += gⱼ₊½           # u[i+1, j+1]
  end

  # return the offset version to make indexing easier
  # (no speed penalty using an offset array here)
  return OffsetMatrix(SMatrix{3,3}(stencil), -1:1, -1:1)
end

function inner_diffusion_operator_2d(edge_metrics, a_edge)
  T = eltype(a_edge)
  # Create a stencil matrix to hold the coefficients for u[i±1,j±1]

  # Don't use an offset matrix here b/c benchmarking showed it was
  # ~3x slower. It makes the indexing here more of a pain,
  # but 3x slower is a big deal for this performance critical section
  stencil = MMatrix{3,3,T}(0, 0, 0, 0, 0, 0, 0, 0, 0)

  # Here the Δξ and Δη terms are 1
  a_ᵢ₊½, a_ᵢ₋½, a_ⱼ₊½, a_ⱼ₋½ = a_edge
  # @unpack αᵢ₊½, αᵢ₋½, αⱼ₊½, αⱼ₋½

  # edge_metrics = metrics_2d(mesh, i, j)
  Jᵢ₊½ = edge_metrics.Jᵢ₊½
  Jᵢ₋½ = edge_metrics.Jᵢ₋½
  Jⱼ₊½ = edge_metrics.Jⱼ₊½
  Jⱼ₋½ = edge_metrics.Jⱼ₋½
  Jξx_ᵢ₊½ = edge_metrics.Jξx_ᵢ₊½
  Jξy_ᵢ₊½ = edge_metrics.Jξy_ᵢ₊½
  Jηx_ᵢ₊½ = edge_metrics.Jηx_ᵢ₊½
  Jηy_ᵢ₊½ = edge_metrics.Jηy_ᵢ₊½
  Jξx_ᵢ₋½ = edge_metrics.Jξx_ᵢ₋½
  Jξy_ᵢ₋½ = edge_metrics.Jξy_ᵢ₋½
  Jηx_ᵢ₋½ = edge_metrics.Jηx_ᵢ₋½
  Jηy_ᵢ₋½ = edge_metrics.Jηy_ᵢ₋½
  Jξx_ⱼ₊½ = edge_metrics.Jξx_ⱼ₊½
  Jξy_ⱼ₊½ = edge_metrics.Jξy_ⱼ₊½
  Jηx_ⱼ₊½ = edge_metrics.Jηx_ⱼ₊½
  Jηy_ⱼ₊½ = edge_metrics.Jηy_ⱼ₊½
  Jξx_ⱼ₋½ = edge_metrics.Jξx_ⱼ₋½
  Jξy_ⱼ₋½ = edge_metrics.Jξy_ⱼ₋½
  Jηx_ⱼ₋½ = edge_metrics.Jηx_ⱼ₋½
  Jηy_ⱼ₋½ = edge_metrics.Jηy_ⱼ₋½

  #------------------------------------------------------------------------------
  # Equations 3.43 and 3.44
  #------------------------------------------------------------------------------
  fᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
  fᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2) / Jᵢ₋½
  # i terms = fᵢ₋½ * u[i-1,j] - (fᵢ₊½ + fᵢ₋½) * u[i,j] + fᵢ₊½ * u[i+1,j]

  stencil[1, 2] += fᵢ₋½           # u[i-1, j]
  stencil[2, 2] += -(fᵢ₊½ + fᵢ₋½) # u[i  , j]
  stencil[3, 2] += fᵢ₊½           # u[i+1, j]

  fⱼ₊½ = a_ⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½
  fⱼ₋½ = a_ⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2) / Jⱼ₋½
  # j terms = fⱼ₋½ * u[i,j-1] - (fⱼ₊½ + fⱼ₋½) * u[i,j] + fⱼ₊½ * u[i,j+1]

  stencil[2, 1] += fⱼ₋½           # u[i, j-1]
  stencil[2, 2] += -(fⱼ₊½ + fⱼ₋½) # u[i, j  ]
  stencil[2, 3] += fⱼ₊½           # u[i, j+1]

  #------------------------------------------------------------------------------
  # cross terms (Equations 3.45 and 3.46)
  #------------------------------------------------------------------------------
  # TODO: pure rectangular grids can skip this section
  gᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½ * Jηx_ᵢ₊½ + Jξy_ᵢ₊½ * Jηy_ᵢ₊½) / (4Jᵢ₊½)
  gᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½ * Jηx_ᵢ₋½ + Jξy_ᵢ₋½ * Jηy_ᵢ₋½) / (4Jᵢ₋½)
  # i terms = gᵢ₊½ * (u[i, j+1] − u[i, j-1] + u[i+1, j+1] − u[i+1, j-1])
  #          -gᵢ₋½ * (u[i, j+1] − u[i, j-1] + u[j−1, j+1] − u[j−1, j-1])

  stencil[1, 1] += gᵢ₋½           # u[i-1, j-1]
  stencil[2, 1] += (-gᵢ₊½ + gᵢ₋½) # u[i  , j-1]
  stencil[3, 1] += -gᵢ₊½          # u[i+1, j-1]
  stencil[1, 3] += -gᵢ₋½          # u[i-1, j+1]
  stencil[2, 3] += (gᵢ₊½ - gᵢ₋½)  # u[i  , j+1]
  stencil[3, 3] += gᵢ₊½           # u[i+1, j+1]

  gⱼ₊½ = a_ⱼ₊½ * (Jξx_ⱼ₊½ * Jηx_ⱼ₊½ + Jξy_ⱼ₊½ * Jηy_ⱼ₊½) / (4Jⱼ₊½)
  gⱼ₋½ = a_ⱼ₋½ * (Jξx_ⱼ₋½ * Jηx_ⱼ₋½ + Jξy_ⱼ₋½ * Jηy_ⱼ₋½) / (4Jⱼ₋½)
  # j terms = gⱼ₊½ * (u[i+1, j] − u[i-1, j] + u[i+1, j+1] − u[i-1, j+1])
  #          -gⱼ₋½ * (u[i+1, j] − u[i-1, j] + u[i+1, j-1] − u[i-1, j-1])

  stencil[1, 1] += gⱼ₋½           # u[i-1, j-1]
  stencil[3, 1] += -gⱼ₋½          # u[i+1, j-1]
  stencil[1, 2] += (-gⱼ₊½ + gⱼ₋½) # u[i-1, j  ]
  stencil[3, 2] += (gⱼ₊½ - gⱼ₋½)  # u[i+1, j  ]
  stencil[1, 3] += -gⱼ₊½          # u[i-1, j+1]
  stencil[3, 3] += gⱼ₊½           # u[i+1, j+1]

  #------------------------------------------------------------------------------
  # RHS term
  #------------------------------------------------------------------------------

  rhs = 1.0
  # return the offset version to make indexing easier
  # (no speed penalty using an offset array here)
  return OffsetMatrix(SMatrix{3,3}(stencil), -1:1, -1:1), rhs
end

function inner_diffusion_operator(uᵢⱼ::T, Jᵢⱼ, sᵢⱼ, Δτ, edge_metrics, a_edge) where {T}
  # Create a stencil matrix to hold the coefficients for u[i±1,j±1]

  # Don't use an offset matrix here b/c benchmarking showed it was
  # ~3x slower. It makes the indexing here more of a pain,
  # but 3x slower is a big deal for this performance critical section

  # Here the Δξ and Δη terms are 1
  a_ᵢ₊½, a_ᵢ₋½, a_ⱼ₊½, a_ⱼ₋½ = a_edge
  # @unpack αᵢ₊½, αᵢ₋½, αⱼ₊½, αⱼ₋½

  # edge_metrics = metrics_2d(mesh, i, j)
  Jᵢ₊½ = edge_metrics.Jᵢ₊½
  Jᵢ₋½ = edge_metrics.Jᵢ₋½
  Jⱼ₊½ = edge_metrics.Jⱼ₊½
  Jⱼ₋½ = edge_metrics.Jⱼ₋½
  Jξx_ᵢ₊½ = edge_metrics.Jξx_ᵢ₊½
  Jξy_ᵢ₊½ = edge_metrics.Jξy_ᵢ₊½
  Jηx_ᵢ₊½ = edge_metrics.Jηx_ᵢ₊½
  Jηy_ᵢ₊½ = edge_metrics.Jηy_ᵢ₊½
  Jξx_ᵢ₋½ = edge_metrics.Jξx_ᵢ₋½
  Jξy_ᵢ₋½ = edge_metrics.Jξy_ᵢ₋½
  Jηx_ᵢ₋½ = edge_metrics.Jηx_ᵢ₋½
  Jηy_ᵢ₋½ = edge_metrics.Jηy_ᵢ₋½
  Jξx_ⱼ₊½ = edge_metrics.Jξx_ⱼ₊½
  Jξy_ⱼ₊½ = edge_metrics.Jξy_ⱼ₊½
  Jηx_ⱼ₊½ = edge_metrics.Jηx_ⱼ₊½
  Jηy_ⱼ₊½ = edge_metrics.Jηy_ⱼ₊½
  Jξx_ⱼ₋½ = edge_metrics.Jξx_ⱼ₋½
  Jξy_ⱼ₋½ = edge_metrics.Jξy_ⱼ₋½
  Jηx_ⱼ₋½ = edge_metrics.Jηx_ⱼ₋½
  Jηy_ⱼ₋½ = edge_metrics.Jηy_ⱼ₋½

  #------------------------------------------------------------------------------
  # Equations 3.43 and 3.44
  #------------------------------------------------------------------------------
  fᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
  fᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2) / Jᵢ₋½
  # i terms = fᵢ₋½ * u[i-1,j] - (fᵢ₊½ + fᵢ₋½) * u[i,j] + fᵢ₊½ * u[i+1,j]

  fⱼ₊½ = a_ⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½
  fⱼ₋½ = a_ⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2) / Jⱼ₋½
  # j terms = fⱼ₋½ * u[i,j-1] - (fⱼ₊½ + fⱼ₋½) * u[i,j] + fⱼ₊½ * u[i,j+1]

  #------------------------------------------------------------------------------
  # cross terms (Equations 3.45 and 3.46)
  #------------------------------------------------------------------------------
  # TODO: pure rectangular grids can skip this section
  gᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½ * Jηx_ᵢ₊½ + Jξy_ᵢ₊½ * Jηy_ᵢ₊½) / (4Jᵢ₊½)
  gᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½ * Jηx_ᵢ₋½ + Jξy_ᵢ₋½ * Jηy_ᵢ₋½) / (4Jᵢ₋½)
  # i terms = gᵢ₊½ * (u[i, j+1] − u[i, j-1] + u[i+1, j+1] − u[i+1, j-1])
  #          -gᵢ₋½ * (u[i, j+1] − u[i, j-1] + u[j−1, j+1] − u[j−1, j-1])

  gⱼ₊½ = a_ⱼ₊½ * (Jξx_ⱼ₊½ * Jηx_ⱼ₊½ + Jξy_ⱼ₊½ * Jηy_ⱼ₊½) / (4Jⱼ₊½)
  gⱼ₋½ = a_ⱼ₋½ * (Jξx_ⱼ₋½ * Jηx_ⱼ₋½ + Jξy_ⱼ₋½ * Jηy_ⱼ₋½) / (4Jⱼ₋½)
  # j terms = gⱼ₊½ * (u[i+1, j] − u[i-1, j] + u[i+1, j+1] − u[i-1, j+1])
  #          -gⱼ₋½ * (u[i+1, j] − u[i-1, j] + u[i+1, j-1] − u[i-1, j-1])

  A = gᵢ₋½ + gⱼ₋½                              # (i-1,j-1)
  B = fⱼ₋½ - gᵢ₊½ + gᵢ₋½                       # (i  ,j-1)
  C = -gᵢ₊½ - gⱼ₋½                             # (i+1,j-1)
  D = fᵢ₋½ - gⱼ₊½ + gⱼ₋½                       # (i-1,j)
  E = -(fᵢ₋½ + fⱼ₋½ + fᵢ₊½ + fⱼ₊½ + Jᵢⱼ / Δτ)  # (i,j)
  F = fᵢ₊½ + gⱼ₊½ - gⱼ₋½                       # (i+1,j)
  G = -gᵢ₋½ - gⱼ₊½                             # (i-1,j+1)
  H = fⱼ₊½ + gᵢ₊½ - gᵢ₋½                       # (i  ,j+1)
  I = gᵢ₊½ + gⱼ₊½                              # (i+1,j+1)
  RHS = -(Jᵢⱼ * sᵢⱼ + uᵢⱼ * Jᵢⱼ / Δτ)

  #------------------------------------------------------------------------------
  # Assemble the stencil
  #------------------------------------------------------------------------------

  stencil = SMatrix{3,3,T}(A, B, C, D, E, F, G, H, I)

  # use an offset so we can index via [+1, -1] for (i+1, j-1)
  offset_stencil = OffsetMatrix(SMatrix{3,3}(stencil), -1:1, -1:1)

  return offset_stencil, RHS
end

function neumann_boundary_diffusion_operator(
  uᵢⱼ::T, Jᵢⱼ, sᵢⱼ, Δτ, edge_metrics, a_edge, loc::Symbol
) where {T}
  # Create a stencil matrix to hold the coefficients for u[i±1,j±1]

  # Don't use an offset matrix here b/c benchmarking showed it was
  # ~3x slower. It makes the indexing here more of a pain,
  # but 3x slower is a big deal for this performance critical section

  # Here the Δξ and Δη terms are 1
  a_ᵢ₊½, a_ᵢ₋½, a_ⱼ₊½, a_ⱼ₋½ = a_edge
  # @unpack αᵢ₊½, αᵢ₋½, αⱼ₊½, αⱼ₋½

  # edge_metrics = metrics_2d(mesh, i, j)
  Jᵢ₊½ = edge_metrics.Jᵢ₊½
  Jᵢ₋½ = edge_metrics.Jᵢ₋½
  Jⱼ₊½ = edge_metrics.Jⱼ₊½
  Jⱼ₋½ = edge_metrics.Jⱼ₋½
  Jξx_ᵢ₊½ = edge_metrics.Jξx_ᵢ₊½
  Jξy_ᵢ₊½ = edge_metrics.Jξy_ᵢ₊½
  Jηx_ᵢ₊½ = edge_metrics.Jηx_ᵢ₊½
  Jηy_ᵢ₊½ = edge_metrics.Jηy_ᵢ₊½
  Jξx_ᵢ₋½ = edge_metrics.Jξx_ᵢ₋½
  Jξy_ᵢ₋½ = edge_metrics.Jξy_ᵢ₋½
  Jηx_ᵢ₋½ = edge_metrics.Jηx_ᵢ₋½
  Jηy_ᵢ₋½ = edge_metrics.Jηy_ᵢ₋½
  Jξx_ⱼ₊½ = edge_metrics.Jξx_ⱼ₊½
  Jξy_ⱼ₊½ = edge_metrics.Jξy_ⱼ₊½
  Jηx_ⱼ₊½ = edge_metrics.Jηx_ⱼ₊½
  Jηy_ⱼ₊½ = edge_metrics.Jηy_ⱼ₊½
  Jξx_ⱼ₋½ = edge_metrics.Jξx_ⱼ₋½
  Jξy_ⱼ₋½ = edge_metrics.Jξy_ⱼ₋½
  Jηx_ⱼ₋½ = edge_metrics.Jηx_ⱼ₋½
  Jηy_ⱼ₋½ = edge_metrics.Jηy_ⱼ₋½

  #------------------------------------------------------------------------------
  # Equations 3.43 and 3.44
  #------------------------------------------------------------------------------
  fᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
  fᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2) / Jᵢ₋½
  # i terms = fᵢ₋½ * u[i-1,j] - (fᵢ₊½ + fᵢ₋½) * u[i,j] + fᵢ₊½ * u[i+1,j]

  fⱼ₊½ = a_ⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½
  fⱼ₋½ = a_ⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2) / Jⱼ₋½
  # j terms = fⱼ₋½ * u[i,j-1] - (fⱼ₊½ + fⱼ₋½) * u[i,j] + fⱼ₊½ * u[i,j+1]

  #------------------------------------------------------------------------------
  # cross terms (Equations 3.45 and 3.46)
  #------------------------------------------------------------------------------
  gᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½ * Jηx_ᵢ₊½ + Jξy_ᵢ₊½ * Jηy_ᵢ₊½) / (4Jᵢ₊½)
  gᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½ * Jηx_ᵢ₋½ + Jξy_ᵢ₋½ * Jηy_ᵢ₋½) / (4Jᵢ₋½)
  # i terms = gᵢ₊½ * (u[i, j+1] − u[i, j-1] + u[i+1, j+1] − u[i+1, j-1])
  #          -gᵢ₋½ * (u[i, j+1] − u[i, j-1] + u[j−1, j+1] − u[j−1, j-1])

  gⱼ₊½ = a_ⱼ₊½ * (Jξx_ⱼ₊½ * Jηx_ⱼ₊½ + Jξy_ⱼ₊½ * Jηy_ⱼ₊½) / (4Jⱼ₊½)
  gⱼ₋½ = a_ⱼ₋½ * (Jξx_ⱼ₋½ * Jηx_ⱼ₋½ + Jξy_ⱼ₋½ * Jηy_ⱼ₋½) / (4Jⱼ₋½)
  # j terms = gⱼ₊½ * (u[i+1, j] − u[i-1, j] + u[i+1, j+1] − u[i-1, j+1])
  #          -gⱼ₋½ * (u[i+1, j] − u[i-1, j] + u[i+1, j-1] − u[i-1, j-1])

  if loc === :ilo
    fᵢ₋½ = zero(T)
    gᵢ₋½ = zero(T)
  elseif loc === :ihi
    fᵢ₊½ = zero(T)
    gᵢ₊½ = zero(T)
  elseif loc === :jlo
    fⱼ₋½ = zero(T)
    gⱼ₋½ = zero(T)
  elseif loc === :jhi
    fⱼ₊½ = zero(T)
    gⱼ₊½ = zero(T)
  end

  A = gᵢ₋½ + gⱼ₋½                              # (i-1,j-1)
  B = fⱼ₋½ - gᵢ₊½ + gᵢ₋½                       # (i  ,j-1)
  C = -gᵢ₊½ - gⱼ₋½                             # (i+1,j-1)
  D = fᵢ₋½ - gⱼ₊½ + gⱼ₋½                       # (i-1,j)
  E = -(fᵢ₋½ + fⱼ₋½ + fᵢ₊½ + fⱼ₊½ + Jᵢⱼ / Δτ)  # (i,j)
  F = fᵢ₊½ + gⱼ₊½ - gⱼ₋½                       # (i+1,j)
  G = -gᵢ₋½ - gⱼ₊½                             # (i-1,j+1)
  H = fⱼ₊½ + gᵢ₊½ - gᵢ₋½                       # (i  ,j+1)
  I = gᵢ₊½ + gⱼ₊½                              # (i+1,j+1)
  RHS = -(Jᵢⱼ * sᵢⱼ + uᵢⱼ * Jᵢⱼ / Δτ)

  #------------------------------------------------------------------------------
  # Assemble the stencil
  #------------------------------------------------------------------------------

  stencil = SMatrix{3,3,T}(A, B, C, D, E, F, G, H, I)

  # use an offset so we can index via [+1, -1] for (i+1, j-1)
  offset_stencil = OffsetMatrix(SMatrix{3,3}(stencil), -1:1, -1:1)

  return offset_stencil, RHS
end

@inline function get_stencil(i, j, u, Δt, scheme)
  uᵢⱼ = u[i, j]
  Jᵢⱼ = scheme.J[i, j]
  sᵢⱼ = scheme.source_term[i, j]
  aᵢ₊½ = scheme.mean_func(scheme.α[i, j], scheme.α[i + 1, j])
  aᵢ₋½ = scheme.mean_func(scheme.α[i, j], scheme.α[i - 1, j])
  aⱼ₊½ = scheme.mean_func(scheme.α[i, j], scheme.α[i, j + 1])
  aⱼ₋½ = scheme.mean_func(scheme.α[i, j], scheme.α[i, j - 1])
  a_edge = (αᵢ₊½=aᵢ₊½, αᵢ₋½=aᵢ₋½, αⱼ₊½=aⱼ₊½, αⱼ₋½=aⱼ₋½)

  edge_metrics = scheme.metrics[i, j]

  stencil, rhs = inner_diffusion_operator(uᵢⱼ, Jᵢⱼ, sᵢⱼ, Δt, edge_metrics, a_edge)
  return stencil, rhs
end

@inline function neumann_boundary_diffusion_operator(i, j, u, Δt, scheme, loc)
  uᵢⱼ = u[i, j]
  Jᵢⱼ = scheme.J[i, j]
  sᵢⱼ = scheme.source_term[i, j]
  aᵢ₊½ = scheme.mean_func(scheme.α[i, j], scheme.α[i + 1, j])
  aᵢ₋½ = scheme.mean_func(scheme.α[i, j], scheme.α[i - 1, j])
  aⱼ₊½ = scheme.mean_func(scheme.α[i, j], scheme.α[i, j + 1])
  aⱼ₋½ = scheme.mean_func(scheme.α[i, j], scheme.α[i, j - 1])
  a_edge = (αᵢ₊½=aᵢ₊½, αᵢ₋½=aᵢ₋½, αⱼ₊½=aⱼ₊½, αⱼ₋½=aⱼ₋½)

  edge_metrics = scheme.metrics[i, j]

  stencil, rhs = neumann_boundary_diffusion_operator(
    uᵢⱼ, Jᵢⱼ, sᵢⱼ, Δt, edge_metrics, a_edge, loc
  )
  return stencil, rhs
end

function assemble_matrix!(scheme::ImplicitScheme, u, Δt)
  ni, nj = size(scheme.domain_indices)
  len = ni * nj
  nhalo = 1
  matrix_domain_LI = LinearIndices(scheme.domain_indices)
  # grid_domain_LI = LinearIndices(scheme.halo_aware_indices)

  matrix_inner_LI = @view matrix_domain_LI[
    (nhalo + 1):(end - nhalo), (nhalo + 1):(end - nhalo)
  ]
  grid_inner_LI = @view scheme.halo_aware_indices[
    (nhalo + 1):(end - nhalo), (nhalo + 1):(end - nhalo)
  ]

  A = scheme.prob.A
  b = scheme.prob.b

  # assemble the inner domain
  for (grid_idx, mat_idx) in zip(grid_inner_LI, matrix_inner_LI)
    i, j = grid_idx.I
    stencil, rhs = get_stencil(i, j, u, Δt, scheme)
    #! format: off
    A[mat_idx, mat_idx - ni - 1] = stencil[-1, -1] # (i-1, j-1)
    A[mat_idx, mat_idx - ni]     = stencil[+0, -1] # (i  , j-1)
    A[mat_idx, mat_idx - ni + 1] = stencil[+1, -1] # (i+1, j-1)
    A[mat_idx, mat_idx - 1]      = stencil[-1, +0] # (i-1, j  )
    A[mat_idx, mat_idx]          = stencil[+0, +0] # (i  , j  )
    A[mat_idx, mat_idx + 1]      = stencil[+1, +0] # (i+1, j  )
    A[mat_idx, mat_idx + ni + 1] = stencil[-1, +1] # (i-1, j+1)
    A[mat_idx, mat_idx + ni]     = stencil[ 0, +1] # (i  , j+1)
    A[mat_idx, mat_idx + ni - 1] = stencil[+1, +1] # (i+1, j+1)
    #! format: on

    b[mat_idx] = rhs
  end

  #------------------------------------------------------------------------------
  # ilo boundary
  #------------------------------------------------------------------------------
  ilo_grid_inner_LI = @view scheme.halo_aware_indices[begin, :]
  ilo_matrix_inner_LI = @view LinearIndices(scheme.domain_indices)[begin, :]

  for (grid_idx, mat_idx) in zip(ilo_grid_inner_LI, ilo_matrix_inner_LI)
    i, j = grid_idx.I
    stencil, rhs = neumann_boundary_diffusion_operator(i, j, u, Δt, scheme, :ilo)

    A[mat_idx, mat_idx] = stencil[+0, +0] # (i, j)
    #! format: off
    # if (1 <= (mat_idx - ni - 1) <= len) A[mat_idx, mat_idx - ni - 1] = stencil[-1, -1] end # (i-1, j-1)
    if (1 <= (mat_idx - ni    ) <= len) A[mat_idx, mat_idx - ni    ] = stencil[+0, -1] end # (i  , j-1)
    if (1 <= (mat_idx - ni + 1) <= len) A[mat_idx, mat_idx - ni + 1] = stencil[+1, -1] end # (i+1, j-1)
    # if (1 <= (mat_idx - 1     ) <= len) A[mat_idx, mat_idx - 1     ] = stencil[-1, +0] end # (i-1, j  )
    if (1 <= (mat_idx + 1     ) <= len) A[mat_idx, mat_idx + 1     ] = stencil[+1, +0] end # (i+1, j  )
    # if (1 <= (mat_idx + ni + 1) <= len) A[mat_idx, mat_idx + ni + 1] = stencil[-1, +1] end # (i-1, j+1)
    if (1 <= (mat_idx + ni    ) <= len) A[mat_idx, mat_idx + ni    ] = stencil[+0, +1] end # (i  , j+1)
    if (1 <= (mat_idx + ni - 1) <= len) A[mat_idx, mat_idx + ni - 1] = stencil[+1, +1] end # (i+1, j+1)   
    #! format: on

    b[mat_idx] = rhs
  end

  #------------------------------------------------------------------------------
  # ihi boundary
  #------------------------------------------------------------------------------
  ihi_grid_inner_LI = @view scheme.halo_aware_indices[end, :]
  ihi_matrix_inner_LI = @view LinearIndices(scheme.domain_indices)[end, :]

  for (grid_idx, mat_idx) in zip(ihi_grid_inner_LI, ihi_matrix_inner_LI)
    i, j = grid_idx.I
    stencil, rhs = neumann_boundary_diffusion_operator(i, j, u, Δt, scheme, :ihi)

    A[mat_idx, mat_idx] = stencil[+0, +0] # (i, j)
    #! format: off
    if (1 <= (mat_idx - ni - 1) <= len) A[mat_idx, mat_idx - ni - 1] = stencil[-1, -1] end # (i-1, j-1)
    if (1 <= (mat_idx - ni    ) <= len) A[mat_idx, mat_idx - ni    ] = stencil[+0, -1] end # (i  , j-1)
    # if (1 <= (mat_idx - ni + 1) <= len) A[mat_idx, mat_idx - ni + 1] = stencil[+1, -1] end # (i+1, j-1)
    if (1 <= (mat_idx - 1     ) <= len) A[mat_idx, mat_idx - 1     ] = stencil[-1, +0] end # (i-1, j  )
    # if (1 <= (mat_idx + 1     ) <= len) A[mat_idx, mat_idx + 1     ] = stencil[+1, +0] end # (i+1, j  )
    if (1 <= (mat_idx + ni + 1) <= len) A[mat_idx, mat_idx + ni + 1] = stencil[-1, +1] end # (i-1, j+1)
    if (1 <= (mat_idx + ni    ) <= len) A[mat_idx, mat_idx + ni    ] = stencil[+0, +1] end # (i  , j+1)
    # if (1 <= (mat_idx + ni - 1) <= len) A[mat_idx, mat_idx + ni - 1] = stencil[+1, +1] end # (i+1, j+1)   
    #! format: on

    b[mat_idx] = rhs
  end

  #------------------------------------------------------------------------------
  # jlo boundary
  #------------------------------------------------------------------------------
  jlo_grid_inner_LI = @view scheme.halo_aware_indices[:, begin]
  jlo_matrix_inner_LI = @view LinearIndices(scheme.domain_indices)[:, begin]

  for (grid_idx, mat_idx) in zip(jlo_grid_inner_LI, jlo_matrix_inner_LI)
    i, j = grid_idx.I
    stencil, rhs = neumann_boundary_diffusion_operator(i, j, u, Δt, scheme, :ihi)

    A[mat_idx, mat_idx] = stencil[+0, +0] # (i, j)
    #! format: off
    # if (1 <= (mat_idx - ni - 1) <= len) A[mat_idx, mat_idx - ni - 1] = stencil[-1, -1] end # (i-1, j-1)
    # if (1 <= (mat_idx - ni    ) <= len) A[mat_idx, mat_idx - ni    ] = stencil[+0, -1] end # (i  , j-1)
    # if (1 <= (mat_idx - ni + 1) <= len) A[mat_idx, mat_idx - ni + 1] = stencil[+1, -1] end # (i+1, j-1)
    if (1 <= (mat_idx - 1     ) <= len) A[mat_idx, mat_idx - 1     ] = stencil[-1, +0] end # (i-1, j  )
    if (1 <= (mat_idx + 1     ) <= len) A[mat_idx, mat_idx + 1     ] = stencil[+1, +0] end # (i+1, j  )
    if (1 <= (mat_idx + ni + 1) <= len) A[mat_idx, mat_idx + ni + 1] = stencil[-1, +1] end # (i-1, j+1)
    if (1 <= (mat_idx + ni    ) <= len) A[mat_idx, mat_idx + ni    ] = stencil[+0, +1] end # (i  , j+1)
    if (1 <= (mat_idx + ni - 1) <= len) A[mat_idx, mat_idx + ni - 1] = stencil[+1, +1] end # (i+1, j+1)   
    #! format: on

    b[mat_idx] = rhs
  end

  #------------------------------------------------------------------------------
  # jhi boundary
  #------------------------------------------------------------------------------
  jhi_grid_inner_LI = @view scheme.halo_aware_indices[:, end]
  jhi_matrix_inner_LI = @view LinearIndices(scheme.domain_indices)[:, end]

  for (grid_idx, mat_idx) in zip(jhi_grid_inner_LI, jhi_matrix_inner_LI)
    i, j = grid_idx.I
    stencil, rhs = neumann_boundary_diffusion_operator(i, j, u, Δt, scheme, :ihi)

    A[mat_idx, mat_idx] = stencil[+0, +0] # (i, j)
    #! format: off
    if (1 <= (mat_idx - ni - 1) <= len) A[mat_idx, mat_idx - ni - 1] = stencil[-1, -1] end # (i-1, j-1)
    if (1 <= (mat_idx - ni    ) <= len) A[mat_idx, mat_idx - ni    ] = stencil[+0, -1] end # (i  , j-1)
    if (1 <= (mat_idx - ni + 1) <= len) A[mat_idx, mat_idx - ni + 1] = stencil[+1, -1] end # (i+1, j-1)
    if (1 <= (mat_idx - 1     ) <= len) A[mat_idx, mat_idx - 1     ] = stencil[-1, +0] end # (i-1, j  )
    if (1 <= (mat_idx + 1     ) <= len) A[mat_idx, mat_idx + 1     ] = stencil[+1, +0] end # (i+1, j  )
    # if (1 <= (mat_idx + ni + 1) <= len) A[mat_idx, mat_idx + ni + 1] = stencil[-1, +1] end # (i-1, j+1)
    # if (1 <= (mat_idx + ni    ) <= len) A[mat_idx, mat_idx + ni    ] = stencil[+0, +1] end # (i  , j+1)
    # if (1 <= (mat_idx + ni - 1) <= len) A[mat_idx, mat_idx + ni - 1] = stencil[+1, +1] end # (i+1, j+1)   
    #! format: on

    b[mat_idx] = rhs
  end

  return nothing
end
