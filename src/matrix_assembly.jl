include("edge_terms.jl")
using KernelAbstractions

"""
    assemble_matrix!(scheme::ImplicitScheme, u, Δt)

Assemble the `A` matrix and right-hand side vector `b` for the solution
to the 2D diffusion problem for a state-array `u` over a time step `Δt`.
"""

function assemble_matrix!(scheme::ImplicitScheme{2}, mesh, u, Δt)
  ni, nj = size(scheme.domain_indices)

  nhalo = 1
  matrix_domain_LI = LinearIndices(scheme.domain_indices)

  matrix_indices = @view matrix_domain_LI[
    (nhalo + 1):(end - nhalo), (nhalo + 1):(end - nhalo)
  ]

  inner_domain = @view scheme.halo_aware_indices[
    (nhalo + 1):(end - nhalo), (nhalo + 1):(end - nhalo)
  ]

  backend = scheme.backend
  workgroup = (64,)

  inner_diffusion_op_kernel_2d!(backend, workgroup)(
    scheme.A,
    scheme.b,
    scheme.α,
    u,
    scheme.source_term,
    Δt,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    inner_domain,
    matrix_indices,
    scheme.mean_func,
    (ni, nj);
    ndrange=size(inner_domain),
  )

  # ilo
  ilo_domain = @view scheme.halo_aware_indices[begin, :]
  ilo_matrix_indices = @view LinearIndices(scheme.domain_indices)[begin, :]
  boundary_diffusion_op_kernel_2d!(backend, workgroup)(
    scheme.A,
    scheme.b,
    scheme.α,
    u,
    scheme.source_term,
    Δt,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    ilo_domain,
    ilo_matrix_indices,
    scheme.mean_func,
    (ni, nj),
    :ilo;
    ndrange=size(ilo_domain),
  )

  # ihi
  ihi_domain = @view scheme.halo_aware_indices[end, :]
  ihi_matrix_indices = @view LinearIndices(scheme.domain_indices)[end, :]
  boundary_diffusion_op_kernel_2d!(backend, workgroup)(
    scheme.A,
    scheme.b,
    scheme.α,
    u,
    scheme.source_term,
    Δt,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    ihi_domain,
    ihi_matrix_indices,
    scheme.mean_func,
    (ni, nj),
    :ihi;
    ndrange=size(ihi_domain),
  )

  # jlo
  jlo_domain = @view scheme.halo_aware_indices[:, begin]
  jlo_matrix_indices = @view LinearIndices(scheme.domain_indices)[:, begin]
  boundary_diffusion_op_kernel_2d!(backend, workgroup)(
    scheme.A,
    scheme.b,
    scheme.α,
    u,
    scheme.source_term,
    Δt,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    jlo_domain,
    jlo_matrix_indices,
    scheme.mean_func,
    (ni, nj),
    :jlo;
    ndrange=size(jlo_domain),
  )

  # jhi
  jhi_domain = @view scheme.halo_aware_indices[:, end]
  jhi_matrix_indices = @view LinearIndices(scheme.domain_indices)[:, end]
  boundary_diffusion_op_kernel_2d!(backend, workgroup)(
    scheme.A,
    scheme.b,
    scheme.α,
    u,
    scheme.source_term,
    Δt,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    jhi_domain,
    jhi_matrix_indices,
    scheme.mean_func,
    (ni, nj),
    :jhi;
    ndrange=size(jhi_domain),
  )

  KernelAbstractions.synchronize(backend)

  return nothing
end

@kernel function inner_diffusion_op_kernel_2d!(
  A,
  b,
  @Const(α),
  @Const(u),
  @Const(source_term),
  @Const(Δt),
  @Const(cell_center_metrics),
  @Const(edge_metrics),
  @Const(grid_indices),
  @Const(matrix_indices),
  @Const(mean_func::F),
  @Const((ni, nj))
) where {F}
  idx = @index(Global, Linear)

  @inbounds begin
    grid_idx = grid_indices[idx]

    i, j = grid_idx.I
    edge_diffusivity = (
      αᵢ₊½=mean_func(α[i, j], α[i + 1, j]),
      αᵢ₋½=mean_func(α[i, j], α[i - 1, j]),
      αⱼ₊½=mean_func(α[i, j], α[i, j + 1]),
      αⱼ₋½=mean_func(α[i, j], α[i, j - 1]),
    )

    stencil, rhs = _inner_diffusion_operator(
      u, source_term, edge_diffusivity, Δt, cell_center_metrics, edge_metrics, grid_idx
    )

    mat_idx = matrix_indices[idx]
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
end

@kernel function boundary_diffusion_op_kernel_2d!(
  A,
  b,
  @Const(α),
  @Const(u),
  @Const(source_term),
  @Const(Δt),
  @Const(cell_center_metrics),
  @Const(edge_metrics),
  @Const(grid_indices),
  @Const(matrix_indices),
  @Const(mean_func::F),
  @Const((ni, nj)),
  @Const(loc::Symbol)
) where {F}
  idx = @index(Global, Linear)

  len = ni * nj

  @inbounds begin
    grid_idx = grid_indices[idx]
    mat_idx = matrix_indices[idx]

    i, j = grid_idx.I
    edge_diffusivity = (
      αᵢ₊½=mean_func(α[i, j], α[i + 1, j]),
      αᵢ₋½=mean_func(α[i, j], α[i - 1, j]),
      αⱼ₊½=mean_func(α[i, j], α[i, j + 1]),
      αⱼ₋½=mean_func(α[i, j], α[i, j - 1]),
    )

    stencil, rhs = _neumann_boundary_diffusion_operator(
      u, source_term, edge_diffusivity, Δt, cell_center_metrics, edge_metrics, grid_idx, loc
    )

    A[mat_idx, mat_idx] = stencil[+0, +0] # (i, j)

    ip1 = true
    jp1 = true
    im1 = true
    jm1 = true
    if loc === :ilo
      im1 = false
    elseif loc === :ihi
      ip1 = false
    elseif loc === :jlo
      jm1 = false
    elseif loc === :jhi
      jp1 = false
    end

    #! format: off
    if ((im1 && jm1) && (1 <= (mat_idx - ni - 1 ) <= len)) A[mat_idx, mat_idx - ni - 1] = stencil[-1, -1] end # (i-1, j-1)
    if ((       jm1) && (1 <= (mat_idx - ni     ) <= len)) A[mat_idx, mat_idx - ni    ] = stencil[+0, -1] end # (i  , j-1)
    if ((im1       ) && (1 <= (mat_idx - 1      ) <= len)) A[mat_idx, mat_idx - 1     ] = stencil[-1, +0] end # (i-1, j  )
    if ((im1 && jp1) && (1 <= (mat_idx + ni - 1 ) <= len)) A[mat_idx, mat_idx + ni - 1] = stencil[-1, +1] end # (i-1, j+1)
    if ((       jp1) && (1 <= (mat_idx + ni     ) <= len)) A[mat_idx, mat_idx + ni    ] = stencil[ 0, +1] end # (i  , j+1)
    if ((ip1 && jm1) && (1 <= (mat_idx - ni + 1 ) <= len)) A[mat_idx, mat_idx - ni + 1] = stencil[+1, -1] end # (i+1, j-1)
    if ((ip1       ) && (1 <= (mat_idx + 1      ) <= len)) A[mat_idx, mat_idx + 1     ] = stencil[+1, +0] end # (i+1, j  )
    if ((ip1 && jp1) && (1 <= (mat_idx + ni + 1 ) <= len)) A[mat_idx, mat_idx + ni + 1] = stencil[+1, +1] end # (i+1, j+1)
    #! format: on

    b[mat_idx] = rhs
  end
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
  u::AbstractArray{T,2},
  source_term::AbstractArray{T,2},
  edge_diffusivity,
  Δτ,
  cell_center_metrics,
  edge_metrics,
  idx::CartesianIndex{2},
) where {T}

  #
  Jᵢⱼ = cell_center_metrics.J[idx]
  sᵢⱼ = source_term[idx]
  uᵢⱼ = u[idx]

  @unpack fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½ = conservative_edge_terms(
    edge_diffusivity, edge_metrics, idx
  )

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

  # Create a stencil matrix to hold the coefficients for u[i±1,j±1]

  # Don't use an offset matrix here b/c benchmarking showed it was
  # ~3x slower. It makes the indexing here more of a pain,
  # but 3x slower is a big deal for this performance critical section

  stencil = SMatrix{3,3,T}(A, B, C, D, E, F, G, H, I)

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
  u::AbstractArray{T,2},
  source_term::AbstractArray{T,2},
  edge_diffusivity,
  Δτ,
  cell_center_metrics,
  edge_metrics,
  idx::CartesianIndex{2},
  loc::Symbol,
) where {T}

  #
  Jᵢⱼ = cell_center_metrics.J[idx]
  sᵢⱼ = source_term[idx]
  uᵢⱼ = u[idx]

  @unpack fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½ = conservative_edge_terms(
    edge_diffusivity, edge_metrics, idx
  )

  # Create a stencil matrix to hold the coefficients for u[i±1,j±1]
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
