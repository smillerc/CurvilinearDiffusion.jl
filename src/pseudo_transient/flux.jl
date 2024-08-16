
@kernel inbounds = true function flux_kernel!(
  qᵢ₊½::AbstractArray{T,N},
  q′ᵢ₊½::AbstractArray{T,N},
  @Const(u),
  @Const(α),
  @Const(θr_dτ),
  @Const(axis),
  @Const(I0),
  mean_func::F,
) where {T,N,F}

  # get the global index and offset for the inner domain
  idx = @index(Global, Cartesian)
  idx += I0

  # ϵ = eps(T)
  @inline ᵢ₊₁ = shift(idx, axis, +1)

  # edge diffusivity / iter params
  @inline αᵢ₊½ = mean_func(α[idx], α[ᵢ₊₁])
  @inline θr_dτ_ᵢ₊½ = mean_func(θr_dτ[idx], θr_dτ[ᵢ₊₁]) # do NOT use max here, or it will fail to converge

  du = u[ᵢ₊₁] - u[idx]
  # du = du * (abs(du) >= ϵ) # perform epsilon check

  _qᵢ₊½ = -αᵢ₊½ * du

  qᵢ₊½[idx] = (qᵢ₊½[idx] * θr_dτ_ᵢ₊½ + _qᵢ₊½) / (1 + θr_dτ_ᵢ₊½)
  q′ᵢ₊½[idx] = _qᵢ₊½
end

function new_flux_kernel!(qᵢ₊½, uᵢ₊₁, uᵢ, αᵢ₊₁, αᵢ, θr_dτᵢ₊₁, θr_dτᵢ)
  harm_mean(a, b) = (2a * b) / (a + b)

  αᵢ₊½ = harm_mean.(αᵢ, αᵢ₊₁)
  θr_dτ_ᵢ₊½ = harm_mean.(θr_dτᵢ, θr_dτᵢ₊₁)

  du = uᵢ₊₁ - uᵢ
  _qᵢ₊½ = -αᵢ₊½ * du

  qᵢ₊½ = (qᵢ₊½ * θr_dτ_ᵢ₊½ + _qᵢ₊½) / (1 + θr_dτ_ᵢ₊½)
  return qᵢ₊½
end

function new_fluxprime_kernel!(uᵢ₊₁, uᵢ, αᵢ₊₁, αᵢ)
  harm_mean(a, b) = (2a * b) / (a + b)

  αᵢ₊½ = harm_mean.(αᵢ, αᵢ₊₁)

  du = uᵢ₊₁ - uᵢ
  return -αᵢ₊½ * du
end

function _cpu_flux_kernel!(
  qᵢ₊½::AbstractArray{T,N}, q′ᵢ₊½, u, α, θr_dτ, axis, domain, mean_func::F
) where {T,N,F}
  #

  ϵ = eps(T)
  @batch for idx in domain
    ᵢ₊₁ = shift(idx, axis, +1)

    # edge diffusivity / iter params
    @inline αᵢ₊½ = mean_func(α[idx], α[ᵢ₊₁])
    @inline θr_dτ_ᵢ₊½ = mean_func(θr_dτ[idx], θr_dτ[ᵢ₊₁]) # do NOT use max here, or it will fail to converge

    du = u[ᵢ₊₁] - u[idx]
    du = du * (abs(du) >= ϵ) # perform epsilon check

    _qᵢ₊½ = -αᵢ₊½ * du

    qᵢ₊½[idx] = (qᵢ₊½[idx] * θr_dτ_ᵢ₊½ + _qᵢ₊½) / (1 + θr_dτ_ᵢ₊½)
    q′ᵢ₊½[idx] = _qᵢ₊½
  end
  return nothing
end

# 2D 
NVTX.@annotate function compute_flux!(
  solver::PseudoTransientSolver{2,T,BE}, ::CurvilinearGrid2D
) where {T,BE<:GPU}
  iaxis, jaxis = (1, 2)

  ᵢ_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)

  ᵢ₊₁_domain = expand_upper(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊₁_domain = expand_upper(solver.iterators.domain.cartesian, jaxis, +1)

  qᵢ′ = @view solver.q′.x[ᵢ_domain]
  qᵢ = @view solver.q.x[ᵢ_domain]
  uᵢ = @view solver.u[ᵢ_domain]
  αᵢ = @view solver.α[ᵢ_domain]
  θr_dτᵢ = @view solver.θr_dτ[ᵢ_domain]
  uᵢ₊₁ = @view solver.u[ᵢ₊₁_domain]
  αᵢ₊₁ = @view solver.α[ᵢ₊₁_domain]
  θr_dτᵢ₊₁ = @view solver.θr_dτ[ᵢ₊₁_domain]

  qᵢ .= map(new_flux_kernel!, qᵢ, uᵢ₊₁, uᵢ, αᵢ₊₁, αᵢ, θr_dτᵢ₊₁, θr_dτᵢ)
  qᵢ′ .= map(new_fluxprime_kernel!, uᵢ₊₁, uᵢ, αᵢ₊₁, αᵢ)

  qⱼ′ = @view solver.q′.y[ⱼ_domain]
  qⱼ = @view solver.q.y[ⱼ_domain]
  uⱼ = @view solver.u[ⱼ_domain]
  αⱼ = @view solver.α[ⱼ_domain]
  θr_dτⱼ = @view solver.θr_dτ[ⱼ_domain]
  uⱼ₊₁ = @view solver.u[ⱼ₊₁_domain]
  αⱼ₊₁ = @view solver.α[ⱼ₊₁_domain]
  θr_dτⱼ₊₁ = @view solver.θr_dτ[ⱼ₊₁_domain]

  qⱼ .= map(new_flux_kernel!, qⱼ, uⱼ₊₁, uⱼ, αⱼ₊₁, αⱼ, θr_dτⱼ₊₁, θr_dτⱼ)
  qⱼ′ .= map(new_fluxprime_kernel!, uⱼ₊₁, uⱼ, αⱼ₊₁, αⱼ)

  return nothing
end

# function compute_flux!(
#   solver::PseudoTransientSolver{2,T,BE}, ::CurvilinearGrid2D
# ) where {T,BE<:GPU}
#   iaxis, jaxis = (1, 2)

#   ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
#   ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)

#   # domain = solver.iterators.domain.cartesian
#   ᵢ₊½_idx_offset = first(ᵢ₊½_domain) - oneunit(first(ᵢ₊½_domain))
#   ⱼ₊½_idx_offset = first(ⱼ₊½_domain) - oneunit(first(ⱼ₊½_domain))

#   @timeit "flux_i" begin
#     flux_kernel!(solver.backend)(
#       solver.q.x,
#       solver.q′.x,
#       solver.u,
#       solver.α,
#       solver.θr_dτ,
#       iaxis,
#       ᵢ₊½_idx_offset,
#       solver.mean;
#       ndrange=size(ᵢ₊½_domain),
#     )

#     KernelAbstractions.synchronize(solver.backend)
#   end

#   @timeit "flux_j" begin
#     flux_kernel!(solver.backend)(
#       solver.q.y,
#       solver.q′.y,
#       solver.u,
#       solver.α,
#       solver.θr_dτ,
#       jaxis,
#       ⱼ₊½_idx_offset,
#       solver.mean;
#       ndrange=size(ⱼ₊½_domain),
#     )

#     KernelAbstractions.synchronize(solver.backend)
#   end
#   return nothing
# end

function compute_flux!(
  solver::PseudoTransientSolver{2,T,BE}, ::CurvilinearGrid2D
) where {T,BE<:CPU}

  #
  iaxis, jaxis = (1, 2)

  ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)

  _cpu_flux_kernel!(
    solver.q.x,
    solver.q′.x,
    solver.u,
    solver.α,
    solver.θr_dτ,
    iaxis,
    ᵢ₊½_domain,
    solver.mean;
  )

  _cpu_flux_kernel!(
    solver.q.y,
    solver.q′.y,
    solver.u,
    solver.α,
    solver.θr_dτ,
    jaxis,
    ⱼ₊½_domain,
    solver.mean;
  )

  return nothing
end

# 3D 

function compute_flux!(
  solver::PseudoTransientSolver{3,T,BE}, ::CurvilinearGrid3D
) where {T,BE<:CPU}
  iaxis, jaxis, kaxis = (1, 2, 3)

  ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)
  ₖ₊½_domain = expand_lower(solver.iterators.domain.cartesian, kaxis, +1)

  _cpu_flux_kernel!(
    solver.q.x,
    solver.q′.x,
    solver.u,
    solver.α,
    solver.θr_dτ,
    iaxis,
    ᵢ₊½_domain,
    solver.mean;
  )

  _cpu_flux_kernel!(
    solver.q.y,
    solver.q′.y,
    solver.u,
    solver.α,
    solver.θr_dτ,
    jaxis,
    ⱼ₊½_domain,
    solver.mean;
  )

  _cpu_flux_kernel!(
    solver.q.z,
    solver.q′.z,
    solver.u,
    solver.α,
    solver.θr_dτ,
    kaxis,
    ₖ₊½_domain,
    solver.mean;
  )

  return nothing
end

function compute_flux!(
  solver::PseudoTransientSolver{3,T,BE}, ::CurvilinearGrid3D
) where {T,BE<:GPU}
  iaxis, jaxis, kaxis = (1, 2, 3)

  ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)
  ₖ₊½_domain = expand_lower(solver.iterators.domain.cartesian, kaxis, +1)

  ᵢ₊½_idx_offset = first(ᵢ₊½_domain) - oneunit(first(ᵢ₊½_domain))
  ⱼ₊½_idx_offset = first(ⱼ₊½_domain) - oneunit(first(ⱼ₊½_domain))
  ₖ₊½_idx_offset = first(ₖ₊½_domain) - oneunit(first(ₖ₊½_domain))

  flux_kernel!(solver.backend)(
    solver.q.x,
    solver.q′.x,
    solver.u,
    solver.α,
    solver.θr_dτ,
    iaxis,
    ᵢ₊½_idx_offset,
    solver.mean;
    ndrange=size(ᵢ₊½_domain),
  )

  flux_kernel!(solver.backend)(
    solver.q.y,
    solver.q′.y,
    solver.u,
    solver.α,
    solver.θr_dτ,
    jaxis,
    ⱼ₊½_idx_offset,
    solver.mean;
    ndrange=size(ⱼ₊½_domain),
  )

  flux_kernel!(solver.backend)(
    solver.q.z,
    solver.q′.z,
    solver.u,
    solver.α,
    solver.θr_dτ,
    kaxis,
    ₖ₊½_idx_offset,
    solver.mean;
    ndrange=size(ₖ₊½_domain),
  )

  KernelAbstractions.synchronize(solver.backend)

  return nothing
end