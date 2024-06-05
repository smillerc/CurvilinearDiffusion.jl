
# TODO: make a linear and non-linear version based on κ or a

"""
The Alternating Direction Explicit (ADE) diffusion solver. Provide the mesh
to give sizes to pre-allocate the type members. Provide a `mean_func` to
tell the solver how to determine diffusivity at the cell edges, i.e. via a
harmonic mean or arithmetic mean.
"""
function ADESolverNSweep(
  mesh::CurvilinearGrid2D,
  bcs;
  face_conductivity::Symbol=:harmonic,
  T=Float64,
  backend=CPU(),
)
  celldims = cellsize_withhalo(mesh)
  uⁿ⁺¹ = zeros(T, celldims)

  nsweeps = 4
  usweepᵏ = ntuple(i -> zeros(T, celldims), nsweeps)

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

  solver = ADESolverNSweep(
    uⁿ⁺¹, usweepᵏ, diffusivity, source_term, mean_func, bcs, iterators, _limits, backend
  )

  return solver
end

"""
# Arguments
 - α: Diffusion coefficient
"""
function solve!(solver::ADESolverNSweep, mesh, u, Δt; cutoff=false)
  @unpack ilo, ihi, jlo, jhi = solver.limits

  domain = mesh.iterators.cell.domain
  copy!(solver.uⁿ⁺¹, u)

  sweep1!(solver.usweepᵏ[1], solver.uⁿ⁺¹, solver, solver.limits, mesh, Δt)
  sweep2!(solver.usweepᵏ[2], solver.uⁿ⁺¹, solver, solver.limits, mesh, Δt)
  sweep3!(solver.usweepᵏ[3], solver.uⁿ⁺¹, solver, solver.limits, mesh, Δt)
  sweep4!(solver.usweepᵏ[4], solver.uⁿ⁺¹, solver, solver.limits, mesh, Δt)

  @inbounds for idx in mesh.iterators.cell.domain
    solver.uⁿ⁺¹[idx] =
      0.25(
        solver.usweepᵏ[1][idx] +
        solver.usweepᵏ[2][idx] +
        solver.usweepᵏ[3][idx] +
        solver.usweepᵏ[4][idx]
      )
  end

  if cutoff
    cutoff!(solver.uⁿ⁺¹)
  end

  # L₂ = residual(u, solver.uⁿ⁺¹, mesh.iterators.cell.domain, Δt)
  @views begin
    L₂ = L₂norm(u[domain], solver.uⁿ⁺¹[domain])
    # L₂ = L₂norm(solver.qⁿ⁺¹[domain], solver.uⁿ⁺¹[domain])
  end

  @printf "\tADESolver L₂: %.1e\n" L₂

  copy!(u, solver.uⁿ⁺¹)

  return nothing
end

# (ilo,jlo) -> (ihi,jhi)
function sweep1!(pⁿ⁺¹, u, solver, limits, mesh, Δt)
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

  CI = CartesianIndices((ilo:ihi, jlo:jhi))

  @inbounds for idx in CI
    i, j = idx.I

    Jᵢⱼ = mesh.cell_center_metrics.J[i, j]
    Js = Jᵢⱼ * solver.source_term[i, j]

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
    #! format: on

    pⁿ⁺¹[i, j] = (
      (
        (Jᵢⱼ / Δt) * pⁿ[i, j] + (
          a_Jξ²ᵢ₊½ * (pⁿ[i + 1, j] - pⁿ[i, j]) +
          a_Jη²ⱼ₊½ * (pⁿ[i, j + 1] - pⁿ[i, j]) +
          a_Jξ²ᵢ₋½ * pⁿ⁺¹[i - 1, j] + # i-1 implicit
          a_Jη²ⱼ₋½ * pⁿ⁺¹[i, j - 1] + # j-1 implicit
          Gᵢ₊½ - Gᵢ₋½ + Gⱼ₊½ - Gⱼ₋½ # non-orthongonal terms at n
          + Js
        )
      ) / (
        (Jᵢⱼ / Δt) +
        a_Jξ²ᵢ₋½ + # i-1 implicit
        a_Jη²ⱼ₋½   # j-1 implicit
      )
    )
  end
end

# (ihi,jhi) -> (ilo,jlo)
function sweep3!(pⁿ⁺¹, u, solver, limits, mesh, Δt)
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

  CI = CartesianIndices((ihi:-1:ilo, jhi:-1:jlo))

  @inbounds for idx in CI
    i, j = idx.I

    Jᵢⱼ = mesh.cell_center_metrics.J[i, j]
    Js = Jᵢⱼ * solver.source_term[i, j]

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
      #! format: on

    pⁿ⁺¹[i, j] = (
      (
        (Jᵢⱼ / Δt) * pⁿ[i, j] + (
          a_Jξ²ᵢ₊½ * pⁿ⁺¹[i + 1, j] + # i+1 implicit
          a_Jη²ⱼ₊½ * pⁿ⁺¹[i, j + 1] - # j+1 implicit
          a_Jξ²ᵢ₋½ * (pⁿ[i, j] - pⁿ[i - 1, j]) - #
          a_Jη²ⱼ₋½ * (pⁿ[i, j] - pⁿ[i, j - 1]) + # current n level
          Gᵢ₊½ - Gᵢ₋½ + Gⱼ₊½ - Gⱼ₋½ # non-orthongonal terms at n
          + Js
        )
      ) / (
        (Jᵢⱼ / Δt) +
        a_Jξ²ᵢ₊½ +  # i+1 implicit
        a_Jη²ⱼ₊½    # j+1 implicit
      )
    )
  end
end

# (ilo,jhi) -> (ihi,jlo)
function sweep2!(pⁿ⁺¹, u, solver, limits, mesh, Δt)
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

  CI = CartesianIndices((ilo:ihi, jhi:-1:jlo))

  @inbounds for idx in CI
    i, j = idx.I

    Jᵢⱼ = mesh.cell_center_metrics.J[i, j]
    Js = Jᵢⱼ * solver.source_term[i, j]

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
    #! format: on

    pⁿ⁺¹[i, j] = (
      (
        (Jᵢⱼ / Δt) * pⁿ[i, j] + (
          a_Jξ²ᵢ₊½ * (pⁿ[i + 1, j] - pⁿ[i, j]) + #
          a_Jη²ⱼ₊½ * (pⁿ⁺¹[i, j + 1]) -  # j+1 implicit
          a_Jξ²ᵢ₋½ * (-pⁿ⁺¹[i - 1, j]) - # i-1 implicit
          a_Jη²ⱼ₋½ * (pⁿ[i, j] - pⁿ[i, j - 1]) + # 
          Gᵢ₊½ - Gᵢ₋½ + Gⱼ₊½ - Gⱼ₋½ # non-orthongonal terms at n
          + Js
        )
      ) / (
        (Jᵢⱼ / Δt) +
        a_Jξ²ᵢ₋½ + # i-1 implicit
        a_Jη²ⱼ₊½   # j+1 implicit
      )
    )
  end
end

# (ihi,jlo) -> (ilo,jhi)
function sweep4!(pⁿ⁺¹, u, solver, limits, mesh, Δt)
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

  CI = CartesianIndices((ihi:-1:ilo, jlo:jhi))

  @inbounds for idx in CI
    i, j = idx.I

    Jᵢⱼ = mesh.cell_center_metrics.J[i, j]
    Js = Jᵢⱼ * solver.source_term[i, j]

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
    #! format: on

    pⁿ⁺¹[i, j] =
      (
        (Jᵢⱼ / Δt) * pⁿ[i, j] + (
          a_Jξ²ᵢ₊½ * (pⁿ⁺¹[i + 1, j]) + # i+1 implicit
          a_Jη²ⱼ₊½ * (pⁿ[i, j + 1] - pⁿ[i, j]) - #
          a_Jξ²ᵢ₋½ * (pⁿ[i, j] - pⁿ[i - 1, j]) - #
          a_Jη²ⱼ₋½ * (-pⁿ⁺¹[i, j - 1]) + # j-1 implicit
          Gᵢ₊½ - Gᵢ₋½ + Gⱼ₊½ - Gⱼ₋½ # non-orthongonal terms at n
          + Js
        )
      ) / (
        (Jᵢⱼ / Δt) +
        a_Jξ²ᵢ₊½ + # i+1 implicit
        a_Jη²ⱼ₋½   # j-1 implicit
      )
  end
end
