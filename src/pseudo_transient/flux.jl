
@kernel function flux_kernel_cons!(
  qᵢ₊½::AbstractArray{T,2},
  q′ᵢ₊½::AbstractArray{T,2},
  H,
  α,
  θr_dτ,
  edge_metrics,
  axis,
  I0,
  mean_func::F,
) where {T,F}
  idx = @index(Global, Cartesian)
  idx += I0

  @inbounds begin
    i, j = idx.I

    ᵢ₊₁ = shift(idx, axis, +1)

    αᵢ₊½ = 0.5(α[i, j] + α[i + 1, j])
    αⱼ₊½ = 0.5(α[i, j] + α[i, j + 1])

    Jᵢ₊½ = edge_metrics.i₊½.J[i, j]
    Jⱼ₊½ = edge_metrics.j₊½.J[i, j]

    Jξx_ᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[i, j]
    Jξy_ᵢ₊½ = edge_metrics.i₊½.ξ̂.x₂[i, j]

    Jηx_ⱼ₊½ = edge_metrics.j₊½.η̂.x₁[i, j]
    Jηy_ⱼ₊½ = edge_metrics.j₊½.η̂.x₂[i, j]

    a_Jξ²ᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½ + Jξy_ᵢ₊½) / Jᵢ₊½
    a_Jη²ⱼ₊½ = αⱼ₊½ * (Jηx_ⱼ₊½ + Jηy_ⱼ₊½) / Jⱼ₊½
    # a_Jξ²ᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
    # a_Jη²ⱼ₊½ = αⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½

    # edge diffusivity / iter params
    # αᵢ₊½ = mean_func(α[idx], α[ᵢ₊₁])
    θr_dτ_ᵢ₊½ = (θr_dτ[idx] + θr_dτ[ᵢ₊₁]) / 2

    # ∇Hᵢ₊½ = mᵢ₊½ * (H[ᵢ₊₁] - H[idx])
    if axis == 1
      _qᵢ₊½ = -a_Jξ²ᵢ₊½ * (H[ᵢ₊₁] - H[idx])
    else
      _qᵢ₊½ = -a_Jη²ⱼ₊½ * (H[ᵢ₊₁] - H[idx])
    end

    qᵢ₊½[idx] = (qᵢ₊½[idx] * θr_dτ_ᵢ₊½ + _qᵢ₊½) / (1 + θr_dτ_ᵢ₊½)
    q′ᵢ₊½[idx] = _qᵢ₊½
  end
end

@kernel function flux_kernel_noncons!(
  qᵢ₊½::AbstractArray{T,2},
  q′ᵢ₊½::AbstractArray{T,2},
  H,
  α,
  θr_dτ,
  cell_center_metrics,
  axis,
  I0,
  mean_func::F,
) where {T,F}
  idx = @index(Global, Cartesian)
  idx += I0

  @inbounds begin
    ᵢ₊₁ = shift(idx, axis, +1)

    # edge diffusivity / iter params
    # αᵢ₊½ = (α[idx] + α[ᵢ₊₁]) / 2
    # θr_dτ_ᵢ₊½ = (θr_dτ[idx] + θr_dτ[ᵢ₊₁]) / 2
    αᵢ₊½ = mean_func(α[idx], α[ᵢ₊₁])
    θr_dτ_ᵢ₊½ = mean_func(θr_dτ[idx], θr_dτ[ᵢ₊₁])

    _qᵢ₊½ = -αᵢ₊½ * (H[ᵢ₊₁] - H[idx])

    qᵢ₊½[idx] = (qᵢ₊½[idx] * θr_dτ_ᵢ₊½ + _qᵢ₊½) / (1 + θr_dτ_ᵢ₊½)
    q′ᵢ₊½[idx] = _qᵢ₊½
  end
end

function compute_flux!(solver::PseudoTransientSolver{2,T}, mesh) where {T}

  #

  iaxis = 1
  jaxis = 2
  ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)

  # domain = solver.iterators.domain.cartesian
  ᵢ₊½_idx_offset = first(ᵢ₊½_domain) - oneunit(first(ᵢ₊½_domain))
  ⱼ₊½_idx_offset = first(ⱼ₊½_domain) - oneunit(first(ⱼ₊½_domain))

  flux_kernel_noncons!(solver.backend)(
    solver.qH.x,
    solver.qH_2.x,
    solver.H,
    solver.α,
    solver.θr_dτ,
    mesh.cell_center_metrics,
    iaxis,
    ᵢ₊½_idx_offset,
    solver.mean;
    ndrange=size(ᵢ₊½_domain),
  )

  flux_kernel_noncons!(solver.backend)(
    solver.qH.y,
    solver.qH_2.y,
    solver.H,
    solver.α,
    solver.θr_dτ,
    mesh.cell_center_metrics,
    jaxis,
    ⱼ₊½_idx_offset,
    solver.mean;
    ndrange=size(ⱼ₊½_domain),
  )

  KernelAbstractions.synchronize(solver.backend)

  return nothing
end

function compute_flux_cons!(solver::PseudoTransientSolver{2,T}, mesh) where {T}

  #

  iaxis = 1
  jaxis = 2
  ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)

  # domain = solver.iterators.domain.cartesian
  ᵢ₊½_idx_offset = first(ᵢ₊½_domain) - oneunit(first(ᵢ₊½_domain))
  ⱼ₊½_idx_offset = first(ⱼ₊½_domain) - oneunit(first(ⱼ₊½_domain))

  flux_kernel_cons!(solver.backend)(
    solver.qH.x,
    solver.qH_2.x,
    solver.H,
    solver.α,
    solver.θr_dτ,
    mesh.edge_metrics,
    iaxis,
    ᵢ₊½_idx_offset,
    solver.mean;
    ndrange=size(ᵢ₊½_domain),
  )

  flux_kernel_cons!(solver.backend)(
    solver.qH.y,
    solver.qH_2.y,
    solver.H,
    solver.α,
    solver.θr_dτ,
    mesh.edge_metrics,
    jaxis,
    ⱼ₊½_idx_offset,
    solver.mean;
    ndrange=size(ⱼ₊½_domain),
  )

  KernelAbstractions.synchronize(solver.backend)

  return nothing
end
