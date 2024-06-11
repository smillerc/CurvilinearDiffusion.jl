
# TODO: make a linear and non-linear version based on κ or a

"""
The Alternating Direction Explicit (ADE) diffusion solver. Provide the mesh
to give sizes to pre-allocate the type members. Provide a `mean_func` to
tell the solver how to determine diffusivity at the cell edges, i.e. via a
harmonic mean or arithmetic mean.
"""
function ADESolver(mesh, bcs; face_conductivity::Symbol=:harmonic, T=Float64, backend=CPU())
  celldims = cellsize_withhalo(mesh)
  uⁿ⁺¹ = KernelAbstractions.zeros(backend, T, celldims)
  qⁿ⁺¹ = KernelAbstractions.zeros(backend, T, celldims)
  pⁿ⁺¹ = KernelAbstractions.zeros(backend, T, celldims)

  diffusivity = KernelAbstractions.zeros(backend, T, celldims)
  source_term = KernelAbstractions.zeros(backend, T, celldims)

  @assert mesh.nhalo >= 1 "The diffusion solver requires the mesh to have a halo region >= 1 cell wide"

  nhalo = 1
  if face_conductivity === :harmonic
    mean_func = harmonic_mean
    @info "Using harmonic mean for face conductivity averaging"
  else
    @info "Using arithmetic mean for face conductivity averaging"
    mean_func = arithmetic_mean
  end

  iterators = (
    domain=(cartesian=mesh.iterators.cell.domain,),
    full=(cartesian=mesh.iterators.cell.full,),
  )

  _limits = limits(iterators.domain.cartesian)

  solver = ADESolver(
    uⁿ⁺¹,
    qⁿ⁺¹,
    pⁿ⁺¹,
    diffusivity,
    source_term,
    mean_func,
    bcs,
    iterators,
    _limits,
    backend,
    nhalo,
  )

  return solver
end

function limits(CI::CartesianIndices{2})
  lo = first.(CI.indices)
  hi = last.(CI.indices)
  return (ilo=lo[1], jlo=lo[2], ihi=hi[1], jhi=hi[2])
end

function limits(CI::CartesianIndices{3})
  lo = first.(CI.indices)
  hi = last.(CI.indices)
  return (ilo=lo[1], jlo=lo[2], klo=lo[3], ihi=hi[1], jhi=hi[2], khi=hi[3])
end

@inline cutoff(a) = (0.5(abs(a) + a)) * isfinite(a)

# function update_diffusivity(ADESolver::solver, κ)
#   @inline for idx in eachindex(solver.diffusivity)
#   end
# end

"""
# Arguments
 - α: Diffusion coefficient
"""
function solve!(
  solver::ADESolver, mesh, u, Δt; cutoff=false, show_convergence=true, kwargs...
)
  domain = mesh.iterators.cell.domain
  copy!(solver.uⁿ⁺¹, u)

  @timeit "reverse_sweep!" reverse_sweep!(
    solver.qⁿ⁺¹, solver.uⁿ⁺¹, solver, solver.limits, mesh, Δt
  )
  @timeit "forward_sweep!" forward_sweep!(
    solver.pⁿ⁺¹, solver.uⁿ⁺¹, solver, solver.limits, mesh, Δt
  )

  @batch for idx in mesh.iterators.cell.domain
    solver.uⁿ⁺¹[idx] = 0.5(solver.qⁿ⁺¹[idx] + solver.pⁿ⁺¹[idx])
  end

  if cutoff
    cutoff!(solver.uⁿ⁺¹)
  end

  # L₂ = residual(u, solver.uⁿ⁺¹, mesh.iterators.cell.domain, Δt)
  @views begin
    # L₂ = L₂norm(u[domain], solver.uⁿ⁺¹[domain])
    L₂ = L₂norm(solver.qⁿ⁺¹[domain], solver.pⁿ⁺¹[domain])
  end

  if show_convergence
    @printf "\tADESolver L₂: %.1e\n" L₂
  end

  @timeit "next_dt" begin
    next_Δt = next_dt(view(solver.uⁿ⁺¹, domain), view(u, domain), Δt, kwargs...)
  end

  copy!(u, solver.uⁿ⁺¹) # update the solution

  return L₂, next_Δt
end

function residual(uⁿ, uⁿ⁺¹, domain, Δt)
  L₂ = 0.0
  N = length(domain)
  @inbounds for idx in domain
    ϵ = abs(uⁿ⁺¹[idx] - uⁿ[idx]) #/ Δt
    L₂ += ϵ^2
  end

  return sqrt(L₂) / N
end

function L₂norm(ϕ1, ϕn)
  denom = sqrt(mapreduce(x -> x^2, +, ϕ1))

  if isinf(denom) || iszero(denom)
    l2norm = -Inf
  else
    f(x, y) = (x - y)^2
    numerator = sqrt(mapreduce(f, +, ϕn, ϕ1))

    l2norm = numerator / denom
  end

  return l2norm
end

# 1D

function forward_sweep!(pⁿ⁺¹::AbstractArray{T,1}, u, solver, limits, mesh, Δt) where {T}
  @unpack ilo, ihi = limits

  domain = solver.iterators.domain.cartesian

  full = expand(domain, solver.nhalo)
  i_bc = haloedge_regions(full, 1, solver.nhalo)

  fill!(pⁿ⁺¹, 0)
  @views begin
    copy!(pⁿ⁺¹[domain], u[domain])
    copy!(pⁿ⁺¹[i_bc.halo.lo], u[i_bc.halo.lo])
    copy!(pⁿ⁺¹[i_bc.halo.hi], u[i_bc.halo.hi])
  end

  # make alias for code readibilty
  pⁿ = pⁿ⁺¹

  # Forward sweep ("implicit" pⁿ⁺¹ for i-1, j-1)
  for i in ilo:ihi
    idx = CartesianIndex(i)
    Jᵢ = mesh.cell_center_metrics.J[i]
    Js = Jᵢ * solver.source_term[i]

    edge_α = edge_diffusivity(solver.α, idx, solver.mean_func)
    edge_terms = conservative_edge_terms(edge_α, mesh.edge_metrics, idx)

    @unpack a_Jξ²ᵢ₊½, a_Jξ²ᵢ₋½ = conservative_edge_terms(edge_terms, mesh.edge_metrics, idx)

    pⁿ⁺¹[i] = (
      (pⁿ[i] + (Δt / Jᵢ) * (a_Jξ²ᵢ₊½ * (pⁿ[i + 1] - pⁿ[i]) + a_Jξ²ᵢ₋½ * pⁿ⁺¹[i - 1] + Js)) /
      (1 + (Δt / Jᵢⱼ) * (a_Jξ²ᵢ₋½))
    )
  end
end

function reverse_sweep!(qⁿ⁺¹::AbstractArray{T,1}, u, solver, limits, mesh, Δt) where {T}
  @unpack ilo, ihi = limits

  domain = solver.iterators.domain.cartesian

  full = expand(domain, solver.nhalo)
  i_bc = haloedge_regions(full, 1, solver.nhalo)

  fill!(qⁿ⁺¹, 0)
  @views begin
    copy!(qⁿ⁺¹[domain], u[domain])
    copy!(qⁿ⁺¹[i_bc.halo.lo], u[i_bc.halo.lo])
    copy!(qⁿ⁺¹[i_bc.halo.hi], u[i_bc.halo.hi])
  end

  # make alias for code readibilty
  qⁿ = qⁿ⁺¹

  # Reverse sweep ("implicit" qⁿ⁺¹ for i+1)
  for i in ihi:-1:ilo
    idx = CartesianIndex(i, j)
    Jᵢⱼ = mesh.cell_center_metrics.J[i]
    Js = Jᵢⱼ * solver.source_term[i]

    edge_α = edge_diffusivity(solver.α, idx, solver.mean_func)
    @unpack a_Jξ²ᵢ₊½, a_Jξ²ᵢ₋½ = conservative_edge_terms(edge_α, mesh.edge_metrics, idx)

    qⁿ⁺¹[i] = (
      (
        qⁿ[i] + (Δt / Jᵢ) * (-a_Jξ²ᵢ₋½ * (qⁿ[i] - qⁿ[i - 1]) - a_Jξ²ᵢ₊½ * qⁿ⁺¹[i + 1] + +Js)
      ) / (1 + (Δt / Jᵢ) * (a_Jξ²ᵢ₊½))
    )
  end
end

# 2D

function forward_sweep!(pⁿ⁺¹::AbstractArray{T,2}, u, solver, limits, mesh, Δt) where {T}
  @unpack ilo, ihi, jlo, jhi = limits

  domain = solver.iterators.domain.cartesian

  full = expand(domain, solver.nhalo)
  i_bc = haloedge_regions(full, 1, solver.nhalo)
  j_bc = haloedge_regions(full, 2, solver.nhalo)

  fill!(pⁿ⁺¹, 0)
  @views begin
    copy!(pⁿ⁺¹[domain], u[domain])
    copy!(pⁿ⁺¹[i_bc.halo.lo], u[i_bc.halo.lo])
    copy!(pⁿ⁺¹[i_bc.halo.hi], u[i_bc.halo.hi])
    copy!(pⁿ⁺¹[j_bc.halo.lo], u[j_bc.halo.lo])
    copy!(pⁿ⁺¹[j_bc.halo.hi], u[j_bc.halo.hi])
  end

  # make alias for code readibilty
  pⁿ = pⁿ⁺¹

  # Forward sweep ("implicit" pⁿ⁺¹ for i-1, j-1)
  for j in jlo:jhi
    for i in ilo:ihi
      idx = CartesianIndex(i, j)
      Jᵢⱼ = mesh.cell_center_metrics.J[i, j]
      Js = Jᵢⱼ * solver.source_term[i, j]

      edge_α = edge_diffusivity(solver.α, idx, solver.mean_func)
      @unpack a_Jξ²ᵢ₊½, a_Jξ²ᵢ₋½, a_Jη²ⱼ₊½, a_Jη²ⱼ₋½, a_Jξηᵢ₊½, a_Jξηᵢ₋½, a_Jηξⱼ₊½, a_Jηξⱼ₋½ = conservative_edge_terms(
        edge_α, mesh.edge_metrics, idx
      )

      #! format: off
      # Gᵢ₊½ = a_Jξηᵢ₊½ * (pⁿ[i, j + 1] - pⁿ[i, j - 1] + pⁿ[i + 1, j + 1] - pⁿ[i + 1, j - 1])
      # Gᵢ₋½ = a_Jξηᵢ₋½ * (pⁿ[i, j + 1] - pⁿ[i, j - 1] + pⁿ[i - 1, j + 1] - pⁿ[i - 1, j - 1])
      # Gⱼ₊½ = a_Jηξⱼ₊½ * (pⁿ[i + 1, j] - pⁿ[i - 1, j] + pⁿ[i + 1, j + 1] - pⁿ[i - 1, j + 1])
      # Gⱼ₋½ = a_Jηξⱼ₋½ * (pⁿ[i + 1, j] - pⁿ[i - 1, j] + pⁿ[i + 1, j - 1] - pⁿ[i - 1, j - 1])

      Gᵢ₊½ = a_Jξηᵢ₊½ * (pⁿ[i, j + 1] - pⁿ⁺¹[i, j - 1] +   pⁿ[i + 1, j + 1] - pⁿ⁺¹[i + 1, j - 1])
      Gᵢ₋½ = a_Jξηᵢ₋½ * (pⁿ[i, j + 1] - pⁿ⁺¹[i, j - 1] + pⁿ⁺¹[i - 1, j + 1] - pⁿ⁺¹[i - 1, j - 1])
      Gⱼ₊½ = a_Jηξⱼ₊½ * (pⁿ[i + 1, j] - pⁿ⁺¹[i - 1, j] +   pⁿ[i + 1, j + 1] - pⁿ⁺¹[i - 1, j + 1])
      Gⱼ₋½ = a_Jηξⱼ₋½ * (pⁿ[i + 1, j] - pⁿ⁺¹[i - 1, j] + pⁿ⁺¹[i + 1, j - 1] - pⁿ⁺¹[i - 1, j - 1])
      #! format: on

      pⁿ⁺¹[i, j] = (
        (
          pⁿ[i, j] +
          (Δt / Jᵢⱼ) * (
            a_Jξ²ᵢ₊½ * (pⁿ[i + 1, j] - pⁿ[i, j]) +
            a_Jη²ⱼ₊½ * (pⁿ[i, j + 1] - pⁿ[i, j]) + # current n level
            a_Jξ²ᵢ₋½ * pⁿ⁺¹[i - 1, j] +
            a_Jη²ⱼ₋½ * pⁿ⁺¹[i, j - 1] + # n+1 level
            Gᵢ₊½ - Gᵢ₋½ + Gⱼ₊½ - Gⱼ₋½ # non-orthongonal terms at n
            + Js
          )
        ) / (1 + (Δt / Jᵢⱼ) * (a_Jξ²ᵢ₋½ + a_Jη²ⱼ₋½))
      )
    end
  end
end

function reverse_sweep!(qⁿ⁺¹::AbstractArray{T,2}, u, solver, limits, mesh, Δt) where {T}
  @unpack ilo, ihi, jlo, jhi = limits

  domain = solver.iterators.domain.cartesian

  full = expand(domain, solver.nhalo)
  i_bc = haloedge_regions(full, 1, solver.nhalo)
  j_bc = haloedge_regions(full, 2, solver.nhalo)

  fill!(qⁿ⁺¹, 0)
  @views begin
    copy!(qⁿ⁺¹[domain], u[domain])
    copy!(qⁿ⁺¹[i_bc.halo.lo], u[i_bc.halo.lo])
    copy!(qⁿ⁺¹[i_bc.halo.hi], u[i_bc.halo.hi])
    copy!(qⁿ⁺¹[j_bc.halo.lo], u[j_bc.halo.lo])
    copy!(qⁿ⁺¹[j_bc.halo.hi], u[j_bc.halo.hi])
  end

  # make alias for code readibilty
  qⁿ = qⁿ⁺¹

  # Reverse sweep ("implicit" pⁿ⁺¹ for i+1, j+1)
  for j in jhi:-1:jlo
    for i in ihi:-1:ilo
      idx = CartesianIndex(i, j)
      Jᵢⱼ = mesh.cell_center_metrics.J[i, j]
      Js = Jᵢⱼ * solver.source_term[i, j]

      edge_α = edge_diffusivity(solver.α, idx, solver.mean_func)
      @unpack a_Jξ²ᵢ₊½, a_Jξ²ᵢ₋½, a_Jη²ⱼ₊½, a_Jη²ⱼ₋½, a_Jξηᵢ₊½, a_Jξηᵢ₋½, a_Jηξⱼ₊½, a_Jηξⱼ₋½ = conservative_edge_terms(
        edge_α, mesh.edge_metrics, idx
      )

      #! format: off
      # Gᵢ₊½ = a_Jξηᵢ₊½ * (qⁿ[i, j + 1] - qⁿ[i, j - 1] + qⁿ[i + 1, j + 1] - qⁿ[i + 1, j - 1])
      # Gᵢ₋½ = a_Jξηᵢ₋½ * (qⁿ[i, j + 1] - qⁿ[i, j - 1] + qⁿ[i - 1, j + 1] - qⁿ[i - 1, j - 1])
      # Gⱼ₊½ = a_Jηξⱼ₊½ * (qⁿ[i + 1, j] - qⁿ[i - 1, j] + qⁿ[i + 1, j + 1] - qⁿ[i - 1, j + 1])
      # Gⱼ₋½ = a_Jηξⱼ₋½ * (qⁿ[i + 1, j] - qⁿ[i - 1, j] + qⁿ[i + 1, j - 1] - qⁿ[i - 1, j - 1])

      Gᵢ₊½ = a_Jξηᵢ₊½ * (qⁿ⁺¹[i, j + 1] - qⁿ[i, j - 1] + qⁿ⁺¹[i + 1, j + 1] - qⁿ⁺¹[i + 1, j - 1])
      Gᵢ₋½ = a_Jξηᵢ₋½ * (qⁿ⁺¹[i, j + 1] - qⁿ[i, j - 1] + qⁿ⁺¹[i - 1, j + 1] - qⁿ[i - 1, j - 1])
      Gⱼ₊½ = a_Jηξⱼ₊½ * (qⁿ⁺¹[i + 1, j] - qⁿ[i - 1, j] + qⁿ⁺¹[i + 1, j + 1] - qⁿ⁺¹[i - 1, j + 1])
      Gⱼ₋½ = a_Jηξⱼ₋½ * (qⁿ⁺¹[i + 1, j] - qⁿ[i - 1, j] + qⁿ⁺¹[i + 1, j - 1] - qⁿ[i - 1, j - 1])
      #! format: on

      qⁿ⁺¹[i, j] = (
        (
          qⁿ[i, j] +
          (Δt / Jᵢⱼ) * (
            -a_Jξ²ᵢ₋½ * (qⁿ[i, j] - qⁿ[i - 1, j]) - #
            a_Jη²ⱼ₋½ * (qⁿ[i, j] - qⁿ[i, j - 1]) +  # current n level
            a_Jξ²ᵢ₊½ * qⁿ⁺¹[i + 1, j] + #
            a_Jη²ⱼ₊½ * qⁿ⁺¹[i, j + 1] + # n+1 level
            Gᵢ₊½ - Gᵢ₋½ + Gⱼ₊½ - Gⱼ₋½ # non-orthongonal terms at n
            + Js
          )
        ) / (1 + (Δt / Jᵢⱼ) * (a_Jξ²ᵢ₊½ + a_Jη²ⱼ₊½))
      )
    end
  end
end

# 3D

function forward_sweep!(pⁿ⁺¹::AbstractArray{T,3}, u, solver, limits, mesh, Δt) where {T}
  @unpack ilo, ihi, jlo, jhi, klo, khi = limits

  domain = solver.iterators.domain.cartesian

  full = expand(domain, solver.nhalo)
  i_bc = haloedge_regions(full, 1, solver.nhalo)
  j_bc = haloedge_regions(full, 2, solver.nhalo)
  k_bc = haloedge_regions(full, 3, solver.nhalo)

  fill!(pⁿ⁺¹, 0)
  @views begin
    copy!(pⁿ⁺¹[domain], u[domain])
    copy!(pⁿ⁺¹[i_bc.halo.lo], u[i_bc.halo.lo])
    copy!(pⁿ⁺¹[i_bc.halo.hi], u[i_bc.halo.hi])
    copy!(pⁿ⁺¹[j_bc.halo.lo], u[j_bc.halo.lo])
    copy!(pⁿ⁺¹[j_bc.halo.hi], u[j_bc.halo.hi])
    copy!(pⁿ⁺¹[k_bc.halo.lo], u[k_bc.halo.lo])
    copy!(pⁿ⁺¹[k_bc.halo.hi], u[k_bc.halo.hi])
  end

  # make alias for code readibilty
  pⁿ = pⁿ⁺¹

  # Forward sweep ("implicit" pⁿ⁺¹ for i-1, j-1)
  for k in klo:khi
    for j in jlo:jhi
      for i in ilo:ihi
        idx = CartesianIndex(i, j, k)
        Jᵢⱼₖ = mesh.cell_center_metrics.J[i, j, k]
        Js = Jᵢⱼₖ * solver.source_term[i, j, k]

        edge_α = edge_diffusivity(solver.α, idx, solver.mean_func)
        edge_terms = conservative_edge_terms(edge_α, mesh.edge_metrics, idx)

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

        #! format: off
        non_orth = (
          0.25a_Jζηₖ₊½ * (pⁿ⁺¹[i, j + 1, k]   + pⁿ[i, j + 1, k + 1]) - #
          0.25a_Jζηₖ₊½ * (  pⁿ[i, j - 1, k]   + pⁿ[i, j - 1, k + 1]) - #
          0.25a_Jζηₖ₋½ * (pⁿ⁺¹[i, j + 1, k]   + pⁿ[i, j + 1, k - 1]) + #
          0.25a_Jζηₖ₋½ * (  pⁿ[i, j - 1, k]   + pⁿ[i, j - 1, k - 1]) + #
          0.25a_Jζξₖ₊½ * (pⁿ⁺¹[i + 1, j, k]   + pⁿ[i + 1, j, k + 1]) - #
          0.25a_Jζξₖ₊½ * (  pⁿ[i - 1, j, k]   + pⁿ[i - 1, j, k + 1]) - #
          0.25a_Jζξₖ₋½ * (pⁿ⁺¹[i + 1, j, k]   + pⁿ[i + 1, j, k - 1]) + #
          0.25a_Jζξₖ₋½ * (  pⁿ[i - 1, j, k]   + pⁿ[i - 1, j, k - 1]) + #
          0.25a_Jηζⱼ₊½ * (pⁿ[i, j + 1, k + 1] + pⁿ⁺¹[i, j, k + 1]) - #
          0.25a_Jηζⱼ₊½ * (pⁿ[i, j + 1, k - 1] +   pⁿ[i, j, k - 1]) - #
          0.25a_Jηζⱼ₋½ * (pⁿ[i, j - 1, k + 1] + pⁿ⁺¹[i, j, k + 1]) + #
          0.25a_Jηζⱼ₋½ * (pⁿ[i, j - 1, k - 1] +   pⁿ[i, j, k - 1]) + #
          0.25a_Jηξⱼ₊½ * (pⁿ[i + 1, j + 1, k] + pⁿ⁺¹[i + 1, j, k]) - #
          0.25a_Jηξⱼ₊½ * (pⁿ[i - 1, j + 1, k] +   pⁿ[i - 1, j, k]) - #
          0.25a_Jηξⱼ₋½ * (pⁿ[i + 1, j - 1, k] + pⁿ⁺¹[i + 1, j, k]) + #
          0.25a_Jηξⱼ₋½ * (pⁿ[i - 1, j - 1, k] +   pⁿ[i - 1, j, k]) + #
          0.25a_Jξζᵢ₊½ * (pⁿ[i + 1, j, k + 1] + pⁿ⁺¹[i, j, k + 1]) - #
          0.25a_Jξζᵢ₊½ * (pⁿ[i + 1, j, k - 1] +   pⁿ[i, j, k - 1]) - #
          0.25a_Jξζᵢ₋½ * (pⁿ[i - 1, j, k + 1] + pⁿ⁺¹[i, j, k + 1]) + #
          0.25a_Jξζᵢ₋½ * (pⁿ[i - 1, j, k - 1] +   pⁿ[i, j, k - 1]) + #
          0.25a_Jξηᵢ₊½ * (pⁿ[i + 1, j + 1, k] + pⁿ⁺¹[i, j + 1, k]) - #
          0.25a_Jξηᵢ₊½ * (pⁿ[i + 1, j - 1, k] +   pⁿ[i, j - 1, k]) - #
          0.25a_Jξηᵢ₋½ * (pⁿ[i - 1, j + 1, k] + pⁿ⁺¹[i, j + 1, k]) + #
          0.25a_Jξηᵢ₋½ * (pⁿ[i - 1, j - 1, k] +   pⁿ[i, j - 1, k])   #
        )
        #! format: on

        pⁿᵢⱼₖ = pⁿ[i, j, k]
        pⁿ⁺¹[i, j, k] = (
          (
            pⁿᵢⱼₖ +
            (Δt / Jᵢⱼₖ) * (
              a_Jξ²ᵢ₊½ * (pⁿ[i + 1, j, k] - pⁿᵢⱼₖ) +
              a_Jη²ⱼ₊½ * (pⁿ[i, j + 1, k] - pⁿᵢⱼₖ) +
              a_Jζ²ₖ₊½ * (pⁿ[i, j, k + 1] - pⁿᵢⱼₖ) + # current n level
              a_Jξ²ᵢ₋½ * pⁿ⁺¹[i - 1, j, k] +
              a_Jη²ⱼ₋½ * pⁿ⁺¹[i, j - 1, k] +
              a_Jζ²ₖ₋½ * pⁿ⁺¹[i, j, k - 1] + # n+1 level
              non_orth +
              # Gᵢ₊½ - Gᵢ₋½ + Gⱼ₊½ - Gⱼ₋½ # non-orthongonal terms at n
              Js
            )
          ) / (1 + (Δt / Jᵢⱼₖ) * (a_Jξ²ᵢ₋½ + a_Jη²ⱼ₋½ + a_Jζ²ₖ₋½))
        )
      end
    end
  end
end

function reverse_sweep!(qⁿ⁺¹::AbstractArray{T,3}, u, solver, limits, mesh, Δt) where {T}
  @unpack ilo, ihi, jlo, jhi, klo, khi = limits

  domain = solver.iterators.domain.cartesian

  full = expand(domain, solver.nhalo)
  i_bc = haloedge_regions(full, 1, solver.nhalo)
  j_bc = haloedge_regions(full, 2, solver.nhalo)
  k_bc = haloedge_regions(full, 3, solver.nhalo)

  fill!(qⁿ⁺¹, 0)
  @views begin
    copy!(qⁿ⁺¹[domain], u[domain])
    copy!(qⁿ⁺¹[i_bc.halo.lo], u[i_bc.halo.lo])
    copy!(qⁿ⁺¹[i_bc.halo.hi], u[i_bc.halo.hi])
    copy!(qⁿ⁺¹[j_bc.halo.lo], u[j_bc.halo.lo])
    copy!(qⁿ⁺¹[j_bc.halo.hi], u[j_bc.halo.hi])
    copy!(qⁿ⁺¹[k_bc.halo.lo], u[k_bc.halo.lo])
    copy!(qⁿ⁺¹[k_bc.halo.hi], u[k_bc.halo.hi])
  end

  # make alias for code readibilty
  qⁿ = qⁿ⁺¹

  # Reverse sweep ("implicit" pⁿ⁺¹ for i+1, j+1)
  for k in khi:-1:klo
    for j in jhi:-1:jlo
      for i in ihi:-1:ilo
        idx = CartesianIndex(i, j, k)
        Jᵢⱼₖ = mesh.cell_center_metrics.J[i, j, k]
        Js = Jᵢⱼₖ * solver.source_term[i, j, k]

        edge_α = edge_diffusivity(solver.α, idx, solver.mean_func)
        edge_terms = conservative_edge_terms(edge_α, mesh.edge_metrics, idx)

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

        #! format: off
        non_orth = (
          0.25a_Jζηₖ₊½ * (  qⁿ[i, j + 1, k]   + qⁿ[i, j + 1, k + 1]) - #
          0.25a_Jζηₖ₊½ * (qⁿ⁺¹[i, j - 1, k]   + qⁿ[i, j - 1, k + 1]) - #
          0.25a_Jζηₖ₋½ * (  qⁿ[i, j + 1, k]   + qⁿ[i, j + 1, k - 1]) + #
          0.25a_Jζηₖ₋½ * (qⁿ⁺¹[i, j - 1, k]   + qⁿ[i, j - 1, k - 1]) + #
          0.25a_Jζξₖ₊½ * (  qⁿ[i + 1, j, k]   + qⁿ[i + 1, j, k + 1]) - #
          0.25a_Jζξₖ₊½ * (qⁿ⁺¹[i - 1, j, k]   + qⁿ[i - 1, j, k + 1]) - #
          0.25a_Jζξₖ₋½ * (  qⁿ[i + 1, j, k]   + qⁿ[i + 1, j, k - 1]) + #
          0.25a_Jζξₖ₋½ * (qⁿ⁺¹[i - 1, j, k]   + qⁿ[i - 1, j, k - 1]) + #
          0.25a_Jηζⱼ₊½ * (qⁿ[i, j + 1, k + 1] +   qⁿ[i, j, k + 1]) - #
          0.25a_Jηζⱼ₊½ * (qⁿ[i, j + 1, k - 1] + qⁿ⁺¹[i, j, k - 1]) - #
          0.25a_Jηζⱼ₋½ * (qⁿ[i, j - 1, k + 1] +   qⁿ[i, j, k + 1]) + #
          0.25a_Jηζⱼ₋½ * (qⁿ[i, j - 1, k - 1] + qⁿ⁺¹[i, j, k - 1]) + #
          0.25a_Jηξⱼ₊½ * (qⁿ[i + 1, j + 1, k] +   qⁿ[i + 1, j, k]) - #
          0.25a_Jηξⱼ₊½ * (qⁿ[i - 1, j + 1, k] + qⁿ⁺¹[i - 1, j, k]) - #
          0.25a_Jηξⱼ₋½ * (qⁿ[i + 1, j - 1, k] +   qⁿ[i + 1, j, k]) + #
          0.25a_Jηξⱼ₋½ * (qⁿ[i - 1, j - 1, k] + qⁿ⁺¹[i - 1, j, k]) + #
          0.25a_Jξζᵢ₊½ * (qⁿ[i + 1, j, k + 1] +   qⁿ[i, j, k + 1]) - #
          0.25a_Jξζᵢ₊½ * (qⁿ[i + 1, j, k - 1] + qⁿ⁺¹[i, j, k - 1]) - #
          0.25a_Jξζᵢ₋½ * (qⁿ[i - 1, j, k + 1] +   qⁿ[i, j, k + 1]) + #
          0.25a_Jξζᵢ₋½ * (qⁿ[i - 1, j, k - 1] + qⁿ⁺¹[i, j, k - 1]) + #
          0.25a_Jξηᵢ₊½ * (qⁿ[i + 1, j + 1, k] +   qⁿ[i, j + 1, k]) - #
          0.25a_Jξηᵢ₊½ * (qⁿ[i + 1, j - 1, k] + qⁿ⁺¹[i, j - 1, k]) - #
          0.25a_Jξηᵢ₋½ * (qⁿ[i - 1, j + 1, k] +   qⁿ[i, j + 1, k]) + #
          0.25a_Jξηᵢ₋½ * (qⁿ[i - 1, j - 1, k] + qⁿ⁺¹[i, j - 1, k])   #
        )
        #! format: on

        qⁿᵢⱼₖ = qⁿ[i, j, k]
        qⁿ⁺¹[i, j, k] = (
          (
            qⁿᵢⱼₖ +
            (Δt / Jᵢⱼₖ) * (
              a_Jξ²ᵢ₊½ * qⁿ⁺¹[i + 1, j, k] + #
              a_Jη²ⱼ₊½ * qⁿ⁺¹[i, j + 1, k] + #
              a_Jζ²ₖ₊½ * qⁿ⁺¹[i, j, k + 1] - # n+1 level
              a_Jξ²ᵢ₋½ * (qⁿᵢⱼₖ - qⁿ[i - 1, j, k]) - #
              a_Jη²ⱼ₋½ * (qⁿᵢⱼₖ - qⁿ[i, j - 1, k]) - #
              a_Jζ²ₖ₋½ * (qⁿᵢⱼₖ - qⁿ[i, j, k - 1]) + # current n level
              non_orth +
              # Gᵢ₊½ - Gᵢ₋½ + Gⱼ₊½ - Gⱼ₋½ # non-orthongonal terms at n
              Js
            )
          ) / (1 + (Δt / Jᵢⱼₖ) * (a_Jξ²ᵢ₊½ + a_Jη²ⱼ₊½ + a_Jζ²ₖ₊½))
        )
      end
    end
  end
end

@inline function edge_diffusivity(
  α, idx::CartesianIndex{1}, mean_function::F
) where {F<:Function}
  i, = idx.I
  edge_diffusivity = (
    αᵢ₊½=mean_function(α[i], α[i + 1]), #
    αᵢ₋½=mean_function(α[i], α[i - 1]), #
  )

  return edge_diffusivity
end

@inline function edge_diffusivity(
  α, idx::CartesianIndex{2}, mean_function::F
) where {F<:Function}
  i, j = idx.I
  edge_diffusivity = (
    αᵢ₊½=mean_function(α[i, j], α[i + 1, j]),
    αᵢ₋½=mean_function(α[i, j], α[i - 1, j]),
    αⱼ₊½=mean_function(α[i, j], α[i, j + 1]),
    αⱼ₋½=mean_function(α[i, j], α[i, j - 1]),
  )

  return edge_diffusivity
end

@inline function edge_diffusivity(
  α, idx::CartesianIndex{3}, mean_function::F
) where {F<:Function}
  i, j, k = idx.I
  edge_diffusivity = (
    αᵢ₊½=mean_function(α[i, j, k], α[i + 1, j, k]),
    αᵢ₋½=mean_function(α[i, j, k], α[i - 1, j, k]),
    αⱼ₊½=mean_function(α[i, j, k], α[i, j + 1, k]),
    αⱼ₋½=mean_function(α[i, j, k], α[i, j - 1, k]),
    αₖ₊½=mean_function(α[i, j, k], α[i, j, k + 1]),
    αₖ₋½=mean_function(α[i, j, k], α[i, j, k - 1]),
  )

  return edge_diffusivity
end
