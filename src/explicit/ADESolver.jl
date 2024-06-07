
# TODO: make a linear and non-linear version based on κ or a

"""
The Alternating Direction Explicit (ADE) diffusion solver. Provide the mesh
to give sizes to pre-allocate the type members. Provide a `mean_func` to
tell the solver how to determine diffusivity at the cell edges, i.e. via a
harmonic mean or arithmetic mean.
"""
function ADESolver(
  mesh::CurvilinearGrid2D,
  bcs;
  face_conductivity::Symbol=:harmonic,
  T=Float64,
  backend=CPU(),
)
  celldims = cellsize_withhalo(mesh)
  uⁿ⁺¹ = zeros(T, celldims)
  qⁿ⁺¹ = zeros(T, celldims)
  pⁿ⁺¹ = zeros(T, celldims)

  diffusivity = zeros(T, celldims)
  source_term = zeros(T, celldims)

  @assert mesh.nhalo >= 1 "The diffusion solver requires the mesh to have a halo region >= 1 cell wide"

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
    uⁿ⁺¹, qⁿ⁺¹, pⁿ⁺¹, diffusivity, source_term, mean_func, bcs, iterators, _limits, backend
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
function solve!(solver::ADESolver, mesh, u, Δt; cutoff=false)
  @unpack ilo, ihi, jlo, jhi = solver.limits

  domain = mesh.iterators.cell.domain
  copy!(solver.uⁿ⁺¹, u)

  reverse_sweep!(solver.qⁿ⁺¹, solver.uⁿ⁺¹, solver, solver.limits, mesh, Δt)
  forward_sweep!(solver.pⁿ⁺¹, solver.uⁿ⁺¹, solver, solver.limits, mesh, Δt)

  @inbounds for idx in mesh.iterators.cell.domain
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

  @printf "\tADESolver L₂: %.1e\n" L₂

  copy!(u, solver.uⁿ⁺¹)

  return nothing
end

function solve_subcycle!(solver::ADESolver, mesh, u, Δt; cutoff=false)
  @unpack ilo, ihi, jlo, jhi = solver.limits

  domain = mesh.iterators.cell.domain
  copy!(solver.uⁿ⁺¹, u)

  L₂ = Inf
  tol = 1e-3
  cyc_max = 100
  cyc = 0

  while L₂ > tol && cyc < cyc_max
    cyc += 1
    reverse_sweep!(solver.qⁿ⁺¹, solver.uⁿ⁺¹, solver, solver.limits, mesh, Δt)
    forward_sweep!(solver.pⁿ⁺¹, solver.uⁿ⁺¹, solver, solver.limits, mesh, Δt)

    @inbounds for idx in mesh.iterators.cell.domain
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

    @printf "\tADESolver cycle: %i, L₂: %.1e\n" cyc L₂
  end

  copy!(u, solver.uⁿ⁺¹)

  return nothing
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

function forward_sweep!(pⁿ⁺¹, u, solver, limits, mesh, Δt)
  @unpack ilo, ihi, jlo, jhi = limits

  domain = solver.iterators.domain.cartesian
  nhalo = 1
  i_bc = haloedge_regions(domain, 1, nhalo)
  j_bc = haloedge_regions(domain, 2, nhalo)

  @views begin
    copy!(pⁿ⁺¹[domain], u[domain])
    copy!(pⁿ⁺¹[i_bc.halo.lo], u[i_bc.halo.lo])
    copy!(pⁿ⁺¹[i_bc.halo.hi], u[i_bc.halo.hi])
    copy!(pⁿ⁺¹[j_bc.halo.lo], u[j_bc.halo.lo])
    copy!(pⁿ⁺¹[j_bc.halo.hi], u[j_bc.halo.hi])
  end

  # make alias for code readibilty
  pⁿ = @views pⁿ⁺¹

  # Forward sweep ("implicit" pⁿ⁺¹ for i-1, j-1)
  @inbounds for j in jlo:jhi
    for i in ilo:ihi
      idx = CartesianIndex(i, j)
      Jᵢⱼ = mesh.cell_center_metrics.J[i, j]
      Js = Jᵢⱼ * solver.source_term[i, j]
      # sᵢⱼ = solver.source_term[i, j] * Δt

      aᵢ₊½ = solver.mean_func(solver.α[i, j], solver.α[i + 1, j])
      aᵢ₋½ = solver.mean_func(solver.α[i, j], solver.α[i - 1, j])
      aⱼ₊½ = solver.mean_func(solver.α[i, j], solver.α[i, j + 1])
      aⱼ₋½ = solver.mean_func(solver.α[i, j], solver.α[i, j - 1])
      edge_diffusivity = (αᵢ₊½=aᵢ₊½, αᵢ₋½=aᵢ₋½, αⱼ₊½=aⱼ₊½, αⱼ₋½=aⱼ₋½)

      @unpack a_Jξ²ᵢ₊½, a_Jξ²ᵢ₋½, a_Jη²ⱼ₊½, a_Jη²ⱼ₋½, a_Jξηᵢ₊½, a_Jξηᵢ₋½, a_Jηξⱼ₊½, a_Jηξⱼ₋½ = conservative_edge_terms(
        edge_diffusivity, mesh.edge_metrics, idx
      )

      #! format: off
      Gᵢ₊½ = a_Jξηᵢ₊½ * (pⁿ[i, j + 1] - pⁿ[i, j - 1] + pⁿ[i + 1, j + 1] - pⁿ[i + 1, j - 1])
      Gᵢ₋½ = a_Jξηᵢ₋½ * (pⁿ[i, j + 1] - pⁿ[i, j - 1] + pⁿ[i - 1, j + 1] - pⁿ[i - 1, j - 1])
      Gⱼ₊½ = a_Jηξⱼ₊½ * (pⁿ[i + 1, j] - pⁿ[i - 1, j] + pⁿ[i + 1, j + 1] - pⁿ[i - 1, j + 1])
      Gⱼ₋½ = a_Jηξⱼ₋½ * (pⁿ[i + 1, j] - pⁿ[i - 1, j] + pⁿ[i + 1, j - 1] - pⁿ[i - 1, j - 1])

      # Gᵢ₊½ = a_Jξηᵢ₊½ * (pⁿ[i, j + 1] - pⁿ⁺¹[i, j - 1] +   pⁿ[i + 1, j + 1] - pⁿ⁺¹[i + 1, j - 1])
      # Gᵢ₋½ = a_Jξηᵢ₋½ * (pⁿ[i, j + 1] - pⁿ⁺¹[i, j - 1] + pⁿ⁺¹[i - 1, j + 1] - pⁿ⁺¹[i - 1, j - 1])
      # Gⱼ₊½ = a_Jηξⱼ₊½ * (pⁿ[i + 1, j] - pⁿ⁺¹[i - 1, j] +   pⁿ[i + 1, j + 1] - pⁿ⁺¹[i - 1, j + 1])
      # Gⱼ₋½ = a_Jηξⱼ₋½ * (pⁿ[i + 1, j] - pⁿ⁺¹[i - 1, j] + pⁿ⁺¹[i + 1, j - 1] - pⁿ⁺¹[i - 1, j - 1])
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

      if !isfinite(pⁿ⁺¹[i, j])
        @show (i, j)
        @show a_Jξ²ᵢ₊½ a_Jξ²ᵢ₋½ a_Jη²ⱼ₊½ a_Jη²ⱼ₋½ a_Jξηᵢ₊½ a_Jξηᵢ₋½ a_Jηξⱼ₊½ a_Jηξⱼ₋½
        @show Gᵢ₊½ Gᵢ₋½ Gⱼ₊½ Gⱼ₋½
        @show edge_diffusivity
        @show pⁿ[i, j + 1] pⁿ⁺¹[i, j - 1] pⁿ⁺¹[i - 1, j + 1] pⁿ⁺¹[i - 1, j - 1]
        error("not finite!!!")
      end
    end
  end
end

function reverse_sweep!(qⁿ⁺¹, u, solver, limits, mesh, Δt)
  @unpack ilo, ihi, jlo, jhi = limits

  domain = solver.iterators.domain.cartesian
  nhalo = 1
  i_bc = haloedge_regions(domain, 1, nhalo)
  j_bc = haloedge_regions(domain, 2, nhalo)

  @views begin
    copy!(qⁿ⁺¹[domain], u[domain])
    copy!(qⁿ⁺¹[i_bc.halo.lo], u[i_bc.halo.lo])
    copy!(qⁿ⁺¹[i_bc.halo.hi], u[i_bc.halo.hi])
    copy!(qⁿ⁺¹[j_bc.halo.lo], u[j_bc.halo.lo])
    copy!(qⁿ⁺¹[j_bc.halo.hi], u[j_bc.halo.hi])
  end
  qⁿ = @views qⁿ⁺¹

  # Reverse sweep ("implicit" pⁿ⁺¹ for i+1, j+1)
  @inbounds for j in jhi:-1:jlo
    for i in ihi:-1:ilo
      idx = CartesianIndex(i, j)
      Jᵢⱼ = mesh.cell_center_metrics.J[i, j]
      Js = Jᵢⱼ * solver.source_term[i, j]
      # sᵢⱼ = solver.source_term[i, j] * Δt

      aᵢ₊½ = solver.mean_func(solver.α[i, j], solver.α[i + 1, j])
      aᵢ₋½ = solver.mean_func(solver.α[i, j], solver.α[i - 1, j])
      aⱼ₊½ = solver.mean_func(solver.α[i, j], solver.α[i, j + 1])
      aⱼ₋½ = solver.mean_func(solver.α[i, j], solver.α[i, j - 1])
      edge_diffusivity = (αᵢ₊½=aᵢ₊½, αᵢ₋½=aᵢ₋½, αⱼ₊½=aⱼ₊½, αⱼ₋½=aⱼ₋½)

      @unpack a_Jξ²ᵢ₊½, a_Jξ²ᵢ₋½, a_Jη²ⱼ₊½, a_Jη²ⱼ₋½, a_Jξηᵢ₊½, a_Jξηᵢ₋½, a_Jηξⱼ₊½, a_Jηξⱼ₋½ = conservative_edge_terms(
        edge_diffusivity, mesh.edge_metrics, idx
      )

      #! format: off
      Gᵢ₊½ = a_Jξηᵢ₊½ * (qⁿ[i, j + 1] - qⁿ[i, j - 1] + qⁿ[i + 1, j + 1] - qⁿ[i + 1, j - 1])
      Gᵢ₋½ = a_Jξηᵢ₋½ * (qⁿ[i, j + 1] - qⁿ[i, j - 1] + qⁿ[i - 1, j + 1] - qⁿ[i - 1, j - 1])
      Gⱼ₊½ = a_Jηξⱼ₊½ * (qⁿ[i + 1, j] - qⁿ[i - 1, j] + qⁿ[i + 1, j + 1] - qⁿ[i - 1, j + 1])
      Gⱼ₋½ = a_Jηξⱼ₋½ * (qⁿ[i + 1, j] - qⁿ[i - 1, j] + qⁿ[i + 1, j - 1] - qⁿ[i - 1, j - 1])

      # Gᵢ₊½ = a_Jξηᵢ₊½ * (qⁿ⁺¹[i, j + 1] - qⁿ[i, j - 1] + qⁿ⁺¹[i + 1, j + 1] - qⁿ⁺¹[i + 1, j - 1])
      # Gᵢ₋½ = a_Jξηᵢ₋½ * (qⁿ⁺¹[i, j + 1] - qⁿ[i, j - 1] + qⁿ⁺¹[i - 1, j + 1] - qⁿ[i - 1, j - 1])
      # Gⱼ₊½ = a_Jηξⱼ₊½ * (qⁿ⁺¹[i + 1, j] - qⁿ[i - 1, j] + qⁿ⁺¹[i + 1, j + 1] - qⁿ⁺¹[i - 1, j + 1])
      # Gⱼ₋½ = a_Jηξⱼ₋½ * (qⁿ⁺¹[i + 1, j] - qⁿ[i - 1, j] + qⁿ⁺¹[i + 1, j - 1] - qⁿ[i - 1, j - 1])
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

# @inline cutoff(a) = (0.5(abs(a) + a)) * isfinite(a)
