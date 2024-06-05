using TimerOutputs

function nonlinear_thermal_conduction_step!(
  scheme::ImplicitScheme,
  mesh,
  T,
  ρ,
  cₚ,
  κ,
  Δt,
  nsubcycles=1;
  cutoff=true,
  show_convergence=true,
  apply_density_bc=true,
  kwargs...,
)
  for n in 1:nsubcycles
    dt = Δt / n
    @timeit "applybc!" applybcs!(scheme.bcs, mesh, T)

    if apply_density_bc
      @timeit "applybc!" applybcs!(scheme.bcs, mesh, ρ)
    end

    @timeit "update_conductivity!" update_conductivity!(scheme, T, ρ, cₚ, κ)
    @timeit "validate_diffusivity" validate_diffusivity(scheme)
    @timeit "solve!" solve!(
      scheme, mesh, T, dt; cutoff=cutoff, show_convergence=show_convergence, kwargs...
    )
  end
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