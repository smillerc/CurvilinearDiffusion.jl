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
  enforce_positivity=true,
  show_convergence=true,
  apply_density_bc=true,
  kwargs...,
)
  domain = scheme.iterators.domain.cartesian
  nhalo = 1

  for n in 1:nsubcycles
    dt = Δt / n
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
    @timeit "solve!" ImplicitSchemeType.solve!(
      scheme,
      mesh,
      T,
      dt;
      cutoff=enforce_positivity,
      show_convergence=show_convergence,
      kwargs...,
    )
  end
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
  show_convergence=nothing,
  apply_bcs=true,
  CFL=1.0,
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
    L2 = ADESolvers.solve!(solver, mesh, T, Δt; cutoff=enforce_positivity, kwargs...)
  end

  applybcs!(solver.bcs, mesh, T)

  @timeit "next_dt" begin
    next_Δt = next_dt(solver.uⁿ⁺¹, T, mesh, Δt, kwargs...)
  end
  copy!(T, solver.uⁿ⁺¹) # update the solution

  return next_Δt
end
