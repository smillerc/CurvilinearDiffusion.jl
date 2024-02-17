include("edge_terms.jl")

"""
    assemble_matrix!(scheme::ImplicitScheme, u, Δt)

Assemble the `A` matrix and right-hand side vector `b` for the solution
to the 2D diffusion problem for a state-array `u` over a time step `Δt`.
"""
function assemble_matrix!(scheme::ImplicitScheme{2}, mesh, u, Δt)
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

  A = scheme.A
  b = scheme.b

  # assemble the inner domain
  for (grid_idx, mat_idx) in zip(grid_inner_LI, matrix_inner_LI)
    i, j = grid_idx.I
    stencil, rhs = inner_diffusion_operator((i, j), u, Δt, scheme, mesh)

    #! format: off
    A[mat_idx, mat_idx - ni - 1] = stencil[-1, -1] # (i-1, j-1)
    A[mat_idx, mat_idx - ni]     = stencil[+0, -1] # (i  , j-1)
    A[mat_idx, mat_idx - ni + 1] = stencil[+1, -1] # (i+1, j-1)
    A[mat_idx, mat_idx - 1]      = stencil[-1, +0] # (i-1, j  )
    A[mat_idx, mat_idx]          = stencil[+0, +0] # (i  , j  )
    A[mat_idx, mat_idx + 1]      = stencil[+1, +0] # (i+1, j  )
    A[mat_idx, mat_idx + ni - 1] = stencil[-1, +1] # (i-1, j+1)
    A[mat_idx, mat_idx + ni]     = stencil[ 0, +1] # (i  , j+1)
    A[mat_idx, mat_idx + ni + 1] = stencil[+1, +1] # (i+1, j+1)
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
    stencil, rhs = neumann_boundary_diffusion_operator((i, j), u, Δt, scheme, mesh, :ilo)

    A[mat_idx, mat_idx] = stencil[+0, +0] # (i  , j  )
    #! format: off
    if (1 <= (mat_idx - ni     ) <= len) A[mat_idx, mat_idx - ni    ] = stencil[+0, -1] end # (i  , j-1)
    if (1 <= (mat_idx - ni + 1 ) <= len) A[mat_idx, mat_idx - ni + 1] = stencil[+1, -1] end # (i+1, j-1)
    if (1 <= (mat_idx + 1      ) <= len) A[mat_idx, mat_idx + 1     ] = stencil[+1, +0] end # (i+1, j  )
    if (1 <= (mat_idx + ni     ) <= len) A[mat_idx, mat_idx + ni    ] = stencil[ 0, +1] end # (i  , j+1)
    if (1 <= (mat_idx + ni + 1 ) <= len) A[mat_idx, mat_idx + ni + 1] = stencil[+1, +1] end # (i+1, j+1)
    # if (1 <= (mat_idx - ni - 1 ) <= len) A[mat_idx, mat_idx - ni - 1] = stencil[-1, -1] end # (i-1, j-1)
    # if (1 <= (mat_idx - 1      ) <= len) A[mat_idx, mat_idx - 1     ] = stencil[-1, +0] end # (i-1, j  )
    # if (1 <= (mat_idx + ni - 1 ) <= len) A[mat_idx, mat_idx + ni - 1] = stencil[-1, +1] end # (i-1, j+1)
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
    stencil, rhs = neumann_boundary_diffusion_operator((i, j), u, Δt, scheme, mesh, :ihi)

    A[mat_idx, mat_idx] = stencil[+0, +0] # (i  , j  )

    #! format: off
    if (1 <= (mat_idx - ni - 1 ) <= len) A[mat_idx, mat_idx - ni - 1] = stencil[-1, -1] end # (i-1, j-1)
    if (1 <= (mat_idx - ni     ) <= len) A[mat_idx, mat_idx - ni    ] = stencil[+0, -1] end # (i  , j-1)
    if (1 <= (mat_idx - 1      ) <= len) A[mat_idx, mat_idx - 1     ] = stencil[-1, +0] end # (i-1, j  )
    if (1 <= (mat_idx + ni - 1 ) <= len) A[mat_idx, mat_idx + ni - 1] = stencil[-1, +1] end # (i-1, j+1)
    if (1 <= (mat_idx + ni     ) <= len) A[mat_idx, mat_idx + ni    ] = stencil[ 0, +1] end # (i  , j+1)
    # if (1 <= (mat_idx - ni + 1 ) <= len) A[mat_idx, mat_idx - ni + 1] = stencil[+1, -1] end # (i+1, j-1)
    # if (1 <= (mat_idx + 1      ) <= len) A[mat_idx, mat_idx + 1     ] = stencil[+1, +0] end # (i+1, j  )
    # if (1 <= (mat_idx + ni + 1 ) <= len) A[mat_idx, mat_idx + ni + 1] = stencil[+1, +1] end # (i+1, j+1)
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
    stencil, rhs = neumann_boundary_diffusion_operator((i, j), u, Δt, scheme, mesh, :jlo)

    A[mat_idx, mat_idx] = stencil[+0, +0] # (i  , j  )
    #! format: off
    if (1 <= (mat_idx - 1      ) <= len) A[mat_idx, mat_idx - 1     ] = stencil[-1, +0] end # (i-1, j  )
    if (1 <= (mat_idx + 1      ) <= len) A[mat_idx, mat_idx + 1     ] = stencil[+1, +0] end # (i+1, j  )
    if (1 <= (mat_idx + ni - 1 ) <= len) A[mat_idx, mat_idx + ni - 1] = stencil[-1, +1] end # (i-1, j+1)
    if (1 <= (mat_idx + ni     ) <= len) A[mat_idx, mat_idx + ni    ] = stencil[ 0, +1] end # (i  , j+1)
    if (1 <= (mat_idx + ni + 1 ) <= len) A[mat_idx, mat_idx + ni + 1] = stencil[+1, +1] end # (i+1, j+1)
    # if (1 <= (mat_idx - ni - 1 ) <= len) A[mat_idx, mat_idx - ni - 1] = stencil[-1, -1] end # (i-1, j-1)
    # if (1 <= (mat_idx - ni     ) <= len) A[mat_idx, mat_idx - ni    ] = stencil[+0, -1] end # (i  , j-1)
    # if (1 <= (mat_idx - ni + 1 ) <= len) A[mat_idx, mat_idx - ni + 1] = stencil[+1, -1] end # (i+1, j-1)
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
    stencil, rhs = neumann_boundary_diffusion_operator((i, j), u, Δt, scheme, mesh, :jhi)

    A[mat_idx, mat_idx] = stencil[+0, +0] # (i  , j  )
    #! format: off
    if (1 <= (mat_idx - ni - 1 ) <= len) A[mat_idx, mat_idx - ni - 1] = stencil[-1, -1] end # (i-1, j-1)
    if (1 <= (mat_idx - ni     ) <= len) A[mat_idx, mat_idx - ni    ] = stencil[+0, -1] end # (i  , j-1)
    if (1 <= (mat_idx - ni + 1 ) <= len) A[mat_idx, mat_idx - ni + 1] = stencil[+1, -1] end # (i+1, j-1)
    if (1 <= (mat_idx - 1      ) <= len) A[mat_idx, mat_idx - 1     ] = stencil[-1, +0] end # (i-1, j  )
    if (1 <= (mat_idx + 1      ) <= len) A[mat_idx, mat_idx + 1     ] = stencil[+1, +0] end # (i+1, j  )
    # if (1 <= (mat_idx + ni - 1 ) <= len) A[mat_idx, mat_idx + ni - 1] = stencil[-1, +1] end # (i-1, j+1)
    # if (1 <= (mat_idx + ni     ) <= len) A[mat_idx, mat_idx + ni    ] = stencil[ 0, +1] end # (i  , j+1)
    # if (1 <= (mat_idx + ni + 1 ) <= len) A[mat_idx, mat_idx + ni + 1] = stencil[+1, +1] end # (i+1, j+1)
    #! format: on

    b[mat_idx] = rhs
  end

  return nothing
end

@inline function inner_diffusion_operator((i,), u, Δt, scheme, mesh::CurvilinearGrid1D)
  uᵢ = u[i]
  Jᵢ = mesh.cell_center_metrics[i].J
  sᵢ = scheme.source_term[i]
  aᵢ₊½ = scheme.mean_func(scheme.α[i], scheme.α[i + 1])
  aᵢ₋½ = scheme.mean_func(scheme.α[i], scheme.α[i - 1])

  edge_diffusivity = (αᵢ₊½=aᵢ₊½, αᵢ₋½=aᵢ₋½)

  edge_metrics = (i₊½=mesh.edge_metrics.i₊½[i], i₋½=mesh.edge_metrics.i₊½[i - 1])

  stencil, rhs = _inner_diffusion_operator(uᵢ, Jᵢ, sᵢ, Δt, edge_metrics, edge_diffusivity)
  return stencil, rhs
end

@inline function inner_diffusion_operator((i, j), u, Δt, scheme, mesh::CurvilinearGrid2D)
  uᵢⱼ = u[i, j]
  Jᵢⱼ = mesh.cell_center_metrics.J[i, j]
  sᵢⱼ = scheme.source_term[i, j]
  aᵢ₊½ = scheme.mean_func(scheme.α[i, j], scheme.α[i + 1, j])
  aᵢ₋½ = scheme.mean_func(scheme.α[i, j], scheme.α[i - 1, j])
  aⱼ₊½ = scheme.mean_func(scheme.α[i, j], scheme.α[i, j + 1])
  aⱼ₋½ = scheme.mean_func(scheme.α[i, j], scheme.α[i, j - 1])
  edge_diffusivity = (αᵢ₊½=aᵢ₊½, αᵢ₋½=aᵢ₋½, αⱼ₊½=aⱼ₊½, αⱼ₋½=aⱼ₋½)

  edge_metrics = (
    i₊½=mesh.edge_metrics.i₊½[i, j],
    i₋½=mesh.edge_metrics.i₊½[i - 1, j],
    j₊½=mesh.edge_metrics.j₊½[i, j],
    j₋½=mesh.edge_metrics.j₊½[i, j - 1],
  )

  stencil, rhs = _inner_diffusion_operator(
    uᵢⱼ, Jᵢⱼ, sᵢⱼ, Δt, edge_metrics, edge_diffusivity
  )
  return stencil, rhs
end

@inline function inner_diffusion_operator((i, j, k), u, Δt, scheme, mesh::CurvilinearGrid3D)
  uᵢⱼₖ = u[i, j, k]
  Jᵢⱼₖ = mesh.cell_center_metrics[i, j, k].J
  sᵢⱼₖ = scheme.source_term[i, j, k]
  aᵢ₊½ = scheme.mean_func(scheme.α[i, j, k], scheme.α[i + 1, j, k])
  aᵢ₋½ = scheme.mean_func(scheme.α[i, j, k], scheme.α[i - 1, j, k])
  aⱼ₊½ = scheme.mean_func(scheme.α[i, j, k], scheme.α[i, j + 1, k])
  aⱼ₋½ = scheme.mean_func(scheme.α[i, j, k], scheme.α[i, j - 1, k])
  aₖ₊½ = scheme.mean_func(scheme.α[i, j, k], scheme.α[i, j, k + 1])
  aₖ₋½ = scheme.mean_func(scheme.α[i, j, k], scheme.α[i, j, k - 1])
  edge_diffusivity = (αᵢ₊½=aᵢ₊½, αᵢ₋½=aᵢ₋½, αⱼ₊½=aⱼ₊½, αⱼ₋½=aⱼ₋½, αₖ₊½=aₖ₊½, αₖ₋½=aₖ₋½)

  edge_metrics = (
    i₊½=mesh.edge_metrics.i₊½[i, j, k],
    i₋½=mesh.edge_metrics.i₊½[i - 1, j, k],
    j₊½=mesh.edge_metrics.j₊½[i, j, k],
    j₋½=mesh.edge_metrics.j₊½[i, j - 1, k],
    k₊½=mesh.edge_metrics.k₊½[i, j, k],
    k₋½=mesh.edge_metrics.k₊½[i, j, k - 1],
  )

  stencil, rhs = _inner_diffusion_operator(
    uᵢⱼₖ, Jᵢⱼₖ, sᵢⱼₖ, Δt, edge_metrics, edge_diffusivity
  )
  return stencil, rhs
end

# Generate a stencil for a single 1d cell in the interior
@inline function _inner_diffusion_operator(
  uᵢ::T, Jᵢ, sᵢ, Δτ, edge_metrics, edge_diffusivity::ET1D
) where {T}
  # Create a stencil matrix to hold the coefficients for u[i±1]

  @unpack fᵢ₊½, fᵢ₋½ = conservative_edge_terms(edge_diffusivity, edge_metrics)

  #------------------------------------------------------------------------------
  # Equations 3.43 and 3.44
  #------------------------------------------------------------------------------
  # fᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½^2) / Jᵢ₊½
  # fᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½^2) / Jᵢ₋½

  A = fᵢ₋½                       # (i-1)
  B = -(fᵢ₋½ + fᵢ₊½ + Jᵢ / Δτ)  # (i)
  C = fᵢ₊½                       # (i+1)
  RHS = -(Jᵢ * sᵢ + uᵢ * Jᵢ / Δτ)

  #------------------------------------------------------------------------------
  # Assemble the stencil
  #------------------------------------------------------------------------------

  stencil = SVector{3,T}(A, B, C)

  # use an offset so we can index via [+1, -1] for (i+1, j-1)
  offset_stencil = OffsetVector(stencil, -1:1)

  return offset_stencil, RHS
end

# Generate a stencil for a single 2d cell in the interior
@inline function _inner_diffusion_operator(
  uᵢⱼ::T, Jᵢⱼ, sᵢⱼ, Δτ, edge_metrics, edge_diffusivity::ET2D
) where {T}
  # Create a stencil matrix to hold the coefficients for u[i±1,j±1]

  # Don't use an offset matrix here b/c benchmarking showed it was
  # ~3x slower. It makes the indexing here more of a pain,
  # but 3x slower is a big deal for this performance critical section

  @unpack fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½ = conservative_edge_terms(
    edge_diffusivity, edge_metrics
  )

  #------------------------------------------------------------------------------
  # Equations 3.43 and 3.44
  #------------------------------------------------------------------------------
  # fᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
  # fᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2) / Jᵢ₋½
  # # i terms = fᵢ₋½ * u[i-1,j] - (fᵢ₊½ + fᵢ₋½) * u[i,j] + fᵢ₊½ * u[i+1,j]

  # fⱼ₊½ = a_ⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½
  # fⱼ₋½ = a_ⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2) / Jⱼ₋½
  # # j terms = fⱼ₋½ * u[i,j-1] - (fⱼ₊½ + fⱼ₋½) * u[i,j] + fⱼ₊½ * u[i,j+1]

  #------------------------------------------------------------------------------
  # cross terms (Equations 3.45 and 3.46)
  #------------------------------------------------------------------------------
  # gᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½ * Jηx_ᵢ₊½ + Jξy_ᵢ₊½ * Jηy_ᵢ₊½) / (4Jᵢ₊½)
  # gᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½ * Jηx_ᵢ₋½ + Jξy_ᵢ₋½ * Jηy_ᵢ₋½) / (4Jᵢ₋½)
  # # i terms = gᵢ₊½ * (u[i, j+1] − u[i, j-1] + u[i+1, j+1] − u[i+1, j-1])
  # #          -gᵢ₋½ * (u[i, j+1] − u[i, j-1] + u[j−1, j+1] − u[j−1, j-1])

  # gⱼ₊½ = a_ⱼ₊½ * (Jξx_ⱼ₊½ * Jηx_ⱼ₊½ + Jξy_ⱼ₊½ * Jηy_ⱼ₊½) / (4Jⱼ₊½)
  # gⱼ₋½ = a_ⱼ₋½ * (Jξx_ⱼ₋½ * Jηx_ⱼ₋½ + Jξy_ⱼ₋½ * Jηy_ⱼ₋½) / (4Jⱼ₋½)
  # # j terms = gⱼ₊½ * (u[i+1, j] − u[i-1, j] + u[i+1, j+1] − u[i-1, j+1])
  # #          -gⱼ₋½ * (u[i+1, j] − u[i-1, j] + u[i+1, j-1] − u[i-1, j-1])

  A = gᵢ₋½ + gⱼ₋½                              # (i-1,j-1)
  B = fⱼ₋½ - gᵢ₊½ + gᵢ₋½                       # (i  ,j-1)
  C = -gᵢ₊½ - gⱼ₋½                             # (i+1,j-1)
  D = fᵢ₋½ - gⱼ₊½ + gⱼ₋½                       # (i-1,j)
  F = fᵢ₊½ + gⱼ₊½ - gⱼ₋½                       # (i+1,j)
  G = -gᵢ₋½ - gⱼ₊½                             # (i-1,j+1)
  H = fⱼ₊½ + gᵢ₊½ - gᵢ₋½                       # (i  ,j+1)
  I = gᵢ₊½ + gⱼ₊½                              # (i+1,j+1)
  E = -(fᵢ₋½ + fⱼ₋½ + fᵢ₊½ + fⱼ₊½ + Jᵢⱼ / Δτ)  # (i,j)
  RHS = -(Jᵢⱼ * sᵢⱼ + uᵢⱼ * Jᵢⱼ / Δτ)

  #------------------------------------------------------------------------------
  # Assemble the stencil
  #------------------------------------------------------------------------------

  stencil = SMatrix{3,3,T}(A, B, C, D, E, F, G, H, I)
  # @show stencil

  # use an offset so we can index via [+1, -1] for (i+1, j-1)
  offset_stencil = OffsetMatrix(SMatrix{3,3}(stencil), -1:1, -1:1)

  return offset_stencil, RHS
end

# Generate a stencil for a single 3d cell in the interior
@inline function _inner_diffusion_operator(
  uᵢⱼₖ::T, Jᵢⱼₖ, sᵢⱼₖₖ, Δτ, edge_metrics, edge_diffusivity::ET3D
) where {T}
  # Create a stencil matrix to hold the coefficients for u[i±1,j±1]

  # Don't use an offset matrix here b/c benchmarking showed it was
  # ~3x slower. It makes the indexing here more of a pain,
  # but 3x slower is a big deal for this performance critical section

  @unpack fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, fₖ₊½, fₖ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½, gₖ₊½, gₖ₋½ = conservative_edge_terms(
    edge_diffusivity, edge_metrics
  )

  # #------------------------------------------------------------------------------
  # # Equations 3.43 and 3.44
  # #------------------------------------------------------------------------------
  # fᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
  # fᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2) / Jᵢ₋½
  # # i terms = fᵢ₋½ * u[i-1,j] - (fᵢ₊½ + fᵢ₋½) * u[i,j] + fᵢ₊½ * u[i+1,j]

  # fⱼ₊½ = a_ⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½
  # fⱼ₋½ = a_ⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2) / Jⱼ₋½
  # # j terms = fⱼ₋½ * u[i,j-1] - (fⱼ₊½ + fⱼ₋½) * u[i,j] + fⱼ₊½ * u[i,j+1]

  # #------------------------------------------------------------------------------
  # # cross terms (Equations 3.45 and 3.46)
  # #------------------------------------------------------------------------------
  # gᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½ * Jηx_ᵢ₊½ + Jξy_ᵢ₊½ * Jηy_ᵢ₊½) / (4Jᵢ₊½)
  # gᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½ * Jηx_ᵢ₋½ + Jξy_ᵢ₋½ * Jηy_ᵢ₋½) / (4Jᵢ₋½)
  # # i terms = gᵢ₊½ * (u[i, j+1] − u[i, j-1] + u[i+1, j+1] − u[i+1, j-1])
  # #          -gᵢ₋½ * (u[i, j+1] − u[i, j-1] + u[j−1, j+1] − u[j−1, j-1])

  # gⱼ₊½ = a_ⱼ₊½ * (Jξx_ⱼ₊½ * Jηx_ⱼ₊½ + Jξy_ⱼ₊½ * Jηy_ⱼ₊½) / (4Jⱼ₊½)
  # gⱼ₋½ = a_ⱼ₋½ * (Jξx_ⱼ₋½ * Jηx_ⱼ₋½ + Jξy_ⱼ₋½ * Jηy_ⱼ₋½) / (4Jⱼ₋½)
  # # j terms = gⱼ₊½ * (u[i+1, j] − u[i-1, j] + u[i+1, j+1] − u[i-1, j+1])
  # #          -gⱼ₋½ * (u[i+1, j] − u[i-1, j] + u[i+1, j-1] − u[i-1, j-1])

  A = gᵢ₋½ + gⱼ₋½                              # (i-1,j-1)
  B = fⱼ₋½ - gᵢ₊½ + gᵢ₋½                       # (i  ,j-1)
  C = -gᵢ₊½ - gⱼ₋½                             # (i+1,j-1)
  D = fᵢ₋½ - gⱼ₊½ + gⱼ₋½                       # (i-1,j)
  F = fᵢ₊½ + gⱼ₊½ - gⱼ₋½                       # (i+1,j)
  G = -gᵢ₋½ - gⱼ₊½                             # (i-1,j+1)
  H = fⱼ₊½ + gᵢ₊½ - gᵢ₋½                       # (i  ,j+1)
  I = gᵢ₊½ + gⱼ₊½                              # (i+1,j+1)
  E = -(fᵢ₋½ + fⱼ₋½ + fᵢ₊½ + fⱼ₊½ + Jᵢⱼₖ / Δτ)  # (i,j)
  RHS = -(Jᵢⱼₖ * sᵢⱼₖ + uᵢⱼₖ * Jᵢⱼₖ / Δτ)

  #------------------------------------------------------------------------------
  # Assemble the stencil
  #------------------------------------------------------------------------------

  stencil = SMatrix{3,3,T}(A, B, C, D, E, F, G, H, I)

  # use an offset so we can index via [+1, -1] for (i+1, j-1)
  offset_stencil = OffsetMatrix(SMatrix{3,3}(stencil), -1:1, -1:1)

  return offset_stencil, RHS
end

@inline function neumann_boundary_diffusion_operator(
  (i,), u, Δt, scheme, mesh::CurvilinearGrid1D, loc
)
  uᵢ = u[i]
  Jᵢ = mesh.cell_center_metrics[i].J
  sᵢ = scheme.source_term[i]
  aᵢ₊½ = scheme.mean_func(scheme.α[i], scheme.α[i + 1])
  aᵢ₋½ = scheme.mean_func(scheme.α[i], scheme.α[i - 1])

  edge_diffusivity = (αᵢ₊½=aᵢ₊½, αᵢ₋½=aᵢ₋½)

  edge_metrics = (i₊½=mesh.edge_metrics.i₊½[i], i₋½=mesh.edge_metrics.i₊½[i - 1])

  stencil, rhs = _neumann_boundary_diffusion_operator(
    uᵢ, Jᵢ, sᵢ, Δt, edge_metrics, edge_diffusivity, loc
  )
  return stencil, rhs
end

@inline function neumann_boundary_diffusion_operator(
  (i, j), u, Δt, scheme, mesh::CurvilinearGrid2D, loc
)
  uᵢⱼ = u[i, j]
  Jᵢⱼ = mesh.cell_center_metrics[i, j].J
  sᵢⱼ = scheme.source_term[i, j]
  aᵢ₊½ = scheme.mean_func(scheme.α[i, j], scheme.α[i + 1, j])
  aᵢ₋½ = scheme.mean_func(scheme.α[i, j], scheme.α[i - 1, j])
  aⱼ₊½ = scheme.mean_func(scheme.α[i, j], scheme.α[i, j + 1])
  aⱼ₋½ = scheme.mean_func(scheme.α[i, j], scheme.α[i, j - 1])
  edge_diffusivity = (αᵢ₊½=aᵢ₊½, αᵢ₋½=aᵢ₋½, αⱼ₊½=aⱼ₊½, αⱼ₋½=aⱼ₋½)

  edge_metrics = (
    i₊½=mesh.edge_metrics.i₊½[i, j],
    i₋½=mesh.edge_metrics.i₊½[i - 1, j],
    j₊½=mesh.edge_metrics.j₊½[i, j],
    j₋½=mesh.edge_metrics.j₊½[i, j - 1],
  )

  stencil, rhs = _neumann_boundary_diffusion_operator(
    uᵢⱼ, Jᵢⱼ, sᵢⱼ, Δt, edge_metrics, edge_diffusivity, loc
  )
  return stencil, rhs
end

@inline function neumann_boundary_diffusion_operator(
  (i, j, k), u, Δt, scheme, mesh::CurvilinearGrid3D, loc
)
  uᵢⱼₖ = u[i, j, k]
  Jᵢⱼₖ = mesh.cell_center_metrics[i, j, k].J
  sᵢⱼₖ = scheme.source_term[i, j, k]
  aᵢ₊½ = scheme.mean_func(scheme.α[i, j, k], scheme.α[i + 1, j, k])
  aᵢ₋½ = scheme.mean_func(scheme.α[i, j, k], scheme.α[i - 1, j, k])
  aⱼ₊½ = scheme.mean_func(scheme.α[i, j, k], scheme.α[i, j + 1, k])
  aⱼ₋½ = scheme.mean_func(scheme.α[i, j, k], scheme.α[i, j - 1, k])
  aₖ₊½ = scheme.mean_func(scheme.α[i, j, k], scheme.α[i, j, k + 1])
  aₖ₋½ = scheme.mean_func(scheme.α[i, j, k], scheme.α[i, j, k - 1])
  edge_diffusivity = (αᵢ₊½=aᵢ₊½, αᵢ₋½=aᵢ₋½, αⱼ₊½=aⱼ₊½, αⱼ₋½=aⱼ₋½, αₖ₊½=aₖ₊½, αₖ₋½=aₖ₋½)

  edge_metrics = (
    i₊½=mesh.edge_metrics.i₊½[i, j, k],
    i₋½=mesh.edge_metrics.i₊½[i - 1, j, k],
    j₊½=mesh.edge_metrics.j₊½[i, j, k],
    j₋½=mesh.edge_metrics.j₊½[i, j - 1, k],
    k₊½=mesh.edge_metrics.k₊½[i, j, k],
    k₋½=mesh.edge_metrics.k₊½[i, j, k - 1],
  )

  stencil, rhs = _neumann_boundary_diffusion_operator(
    uᵢⱼₖ, Jᵢⱼₖ, sᵢⱼₖ, Δt, edge_metrics, edge_diffusivity, loc
  )
  return stencil, rhs
end

# Generate a stencil for a 1D neumann boundary condition
@inline function _neumann_boundary_diffusion_operator(
  uᵢ::T, Jᵢ, sᵢ, Δτ, edge_metric, edge_diffusivity::ET1D, loc::Symbol
) where {T}
  # Create a stencil matrix to hold the coefficients for u[i±1,j±1]

  @unpack fᵢ₊½, fᵢ₋½ = conservative_edge_terms(edge_diffusivity, edge_metric)

  if loc === :ilo
    fᵢ₋½ = zero(T)
  elseif loc === :ihi
    fᵢ₊½ = zero(T)
  end

  A = fᵢ₋½                     # (i-1,j)
  B = -(fᵢ₋½ + fᵢ₊½ + Jᵢ / Δτ) # (i,j)
  C = fᵢ₊½                     # (i+1,j)
  RHS = -(Jᵢ * sᵢ + uᵢ * Jᵢ / Δτ)

  #------------------------------------------------------------------------------
  # Assemble the stencil
  #------------------------------------------------------------------------------

  stencil = SVector{3,T}(A, B, C)

  # use an offset so we can index via [+1, -1] for (i+1, j-1)
  offset_stencil = OffsetVector(stencil, -1:1)

  return offset_stencil, RHS
end

# Generate a stencil for a 2D neumann boundary condition
@inline function _neumann_boundary_diffusion_operator(
  uᵢⱼ::T, Jᵢⱼ, sᵢⱼ, Δτ, edge_metric, edge_diffusivity::ET2D, loc::Symbol
) where {T}
  # Create a stencil matrix to hold the coefficients for u[i±1,j±1]

  @unpack fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½ = conservative_edge_terms(
    edge_diffusivity, edge_metric
  )

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
  F = fᵢ₊½ + gⱼ₊½ - gⱼ₋½                       # (i+1,j)
  G = -gᵢ₋½ - gⱼ₊½                             # (i-1,j+1)
  H = fⱼ₊½ + gᵢ₊½ - gᵢ₋½                       # (i  ,j+1)
  I = gᵢ₊½ + gⱼ₊½                              # (i+1,j+1)
  E = -(fᵢ₋½ + fⱼ₋½ + fᵢ₊½ + fⱼ₊½ + Jᵢⱼ / Δτ)  # (i,j)
  RHS = -(Jᵢⱼ * sᵢⱼ + uᵢⱼ * Jᵢⱼ / Δτ)

  #------------------------------------------------------------------------------
  # Assemble the stencil
  #------------------------------------------------------------------------------

  stencil = SMatrix{3,3,T}(A, B, C, D, E, F, G, H, I)

  # use an offset so we can index via [+1, -1] for (i+1, j-1)
  offset_stencil = OffsetMatrix(SMatrix{3,3}(stencil), -1:1, -1:1)

  return offset_stencil, RHS
end

# @inline function dirichlet_boundary_diffusion_operator((i,j), u, Δt, scheme, mesh, loc)
#   uᵢⱼ = u[i, j]
#   Jᵢⱼ = scheme.J[i, j]
#   sᵢⱼ = scheme.source_term[i, j]
#   aᵢ₊½ = scheme.mean_func(scheme.α[i, j], scheme.α[i + 1, j])
#   aᵢ₋½ = scheme.mean_func(scheme.α[i, j], scheme.α[i - 1, j])
#   aⱼ₊½ = scheme.mean_func(scheme.α[i, j], scheme.α[i, j + 1])
#   aⱼ₋½ = scheme.mean_func(scheme.α[i, j], scheme.α[i, j - 1])
#   a_edge = (αᵢ₊½=aᵢ₊½, αᵢ₋½=aᵢ₋½, αⱼ₊½=aⱼ₊½, αⱼ₋½=aⱼ₋½)

#   edge_metrics = scheme.metrics[i, j]

#   stencil, rhs = _neumann_boundary_diffusion_operator(
#     uᵢⱼ, Jᵢⱼ, sᵢⱼ, Δt, edge_metrics, a_edge, loc
#   )
#   return stencil, rhs
# end

# # Generate a stencil for a Dirichlet boundary condition
# @inline function _dirichlet_boundary_diffusion_operator(
#   uᵢⱼ::T, Jᵢⱼ, sᵢⱼ, Δτ, edge_metric, a_edge, loc::Symbol
# ) where {T}
#   # Create a stencil matrix to hold the coefficients for u[i±1,j±1]

#   fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½ = conservative_edge_terms(
#     edge_diffusivity, edge_metrics
#   )

#   if loc === :ilo
#     fᵢ₋½ = zero(T)
#     gᵢ₋½ = zero(T)
#   elseif loc === :ihi
#     fᵢ₊½ = zero(T)
#     gᵢ₊½ = zero(T)
#   elseif loc === :jlo
#     fⱼ₋½ = zero(T)
#     gⱼ₋½ = zero(T)
#   elseif loc === :jhi
#     fⱼ₊½ = zero(T)
#     gⱼ₊½ = zero(T)
#   end

#   A = gᵢ₋½ + gⱼ₋½                              # (i-1,j-1)
#   B = fⱼ₋½ - gᵢ₊½ + gᵢ₋½                       # (i  ,j-1)
#   C = -gᵢ₊½ - gⱼ₋½                             # (i+1,j-1)
#   D = fᵢ₋½ - gⱼ₊½ + gⱼ₋½                       # (i-1,j)
#   F = fᵢ₊½ + gⱼ₊½ - gⱼ₋½                       # (i+1,j)
#   G = -gᵢ₋½ - gⱼ₊½                             # (i-1,j+1)
#   H = fⱼ₊½ + gᵢ₊½ - gᵢ₋½                       # (i  ,j+1)
#   I = gᵢ₊½ + gⱼ₊½                              # (i+1,j+1)
#   E = -(fᵢ₋½ + fⱼ₋½ + fᵢ₊½ + fⱼ₊½ + Jᵢⱼ / Δτ)  # (i,j)
#   RHS = -(Jᵢⱼ * sᵢⱼ + uᵢⱼ * Jᵢⱼ / Δτ)

#   #------------------------------------------------------------------------------
#   # Assemble the stencil
#   #------------------------------------------------------------------------------

#   stencil = SMatrix{3,3,T}(A, B, C, D, E, F, G, H, I)

#   # use an offset so we can index via [+1, -1] for (i+1, j-1)
#   offset_stencil = OffsetMatrix(SMatrix{3,3}(stencil), -1:1, -1:1)

#   return offset_stencil, RHS
# end
