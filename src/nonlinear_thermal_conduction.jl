using TimerOutputs
using Printf

function nonlinear_thermal_conduction_step!(
  scheme::ImplicitScheme,
  mesh,
  T,
  ρ,
  cₚ,
  κ,
  Δt;
  enforce_positivity=true,
  show_convergence=true,
  apply_density_bc=true,
  kwargs...,
)
  domain = scheme.iterators.domain.cartesian
  nhalo = 1

  @timeit "applybc!" applybcs!(scheme.bcs, mesh, T)

  if apply_density_bc
    @timeit "applybc!" applybcs!(scheme.bcs, mesh, ρ)
  end

  @timeit "update_conductivity!" update_conductivity!(scheme, mesh, T, ρ, cₚ, κ)

  @timeit "validate_source_term" validate_scalar(
    scheme.source_term, domain, nhalo, :source_term, enforce_positivity=false
  )
  @timeit "validate_diffusivity" validate_scalar(
    scheme.α, domain, nhalo, :diffusivity, enforce_positivity=true
  )
  @timeit "solve!" begin
    stats, next_Δt = ImplicitSchemeType.solve!(
      scheme,
      mesh,
      T,
      Δt;
      cutoff=enforce_positivity,
      show_convergence=show_convergence,
      kwargs...,
    )
  end

  return stats, next_Δt
end

function nonlinear_thermal_conduction_step!(
  solver::PseudoTransientSolver, mesh, T, ρ, cₚ, κ, Δt; show_convergence=true, kwargs...
)
  stats, next_Δt = PseudoTransientScheme.step!(solver, mesh, T, ρ, cₚ, κ, Δt; kwargs...)

  if show_convergence
    @printf "\trel error: %.3e, abs err: %.3e, %i\n" stats.rel_err stats.abs_err stats.niter
  end

  return stats, next_Δt
end

function nonlinear_thermal_conduction_step!(
  solver::AbstractADESolver,
  mesh,
  T,
  ρ,
  cₚ,
  κ,
  Δt;
  enforce_positivity=false,
  show_convergence=true,
  apply_bcs=true,
  kwargs...,
)
  domain = solver.iterators.domain.cartesian
  nhalo = solver.nhalo

  if apply_bcs
    @timeit "applybc!" begin
      applybcs!(solver.bcs, mesh, T)
      applybcs!(solver.bcs, mesh, ρ)
    end
  end

  @timeit "update_conductivity!" update_conductivity!(solver, mesh, T, ρ, cₚ, κ)

  @timeit "validate_source_term" validate_scalar(
    solver.source_term, domain, nhalo, :source_term, enforce_positivity=false
  )
  @timeit "validate_diffusivity" validate_scalar(
    solver.α, domain, nhalo, :diffusivity, enforce_positivity=true
  )

  @timeit "solve!" begin
    stats, next_Δt = ADESolvers.solve!(
      solver,
      mesh,
      T,
      Δt;
      cutoff=enforce_positivity,
      show_convergence=show_convergence,
      kwargs...,
    )
  end

  @timeit "validate_temperature" validate_scalar(
    T, domain, nhalo, :temperature, enforce_positivity=true
  )

  return stats, next_Δt
end

function validate_diffusivity(solver)
  nhalo = 1

  domain = solver.iterators.domain.cartesian

  α_domain = @view solver.α[domain]

  domain_valid = (all(isfinite.(α_domain)) && all(map(x -> x >= 0, α_domain)))

  if !domain_valid
    error("Invalid diffusivity in the domain")
  end

  N = length(size(solver.α))
  for axis in 1:N
    bc = haloedge_regions(domain, axis, nhalo)
    lo_edge = bc.halo.lo
    hi_edge = bc.halo.hi

    α_lo = @view solver.α[lo_edge]
    α_hi = @view solver.α[hi_edge]

    α_lo_valid = (all(isfinite.(α_lo)) && all(map(x -> x >= 0, α_lo)))
    α_hi_valid = (all(isfinite.(α_hi)) && all(map(x -> x >= 0, α_hi)))

    if !α_lo_valid
      error("Invalid diffusivity in the lo halo region for axis: $axis")
    end

    if !α_hi_valid
      error("Invalid diffusivity in the hi halo region for axis: $axis")
    end
  end
end
