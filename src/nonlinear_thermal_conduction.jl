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

    @timeit "update_conductivity!" update_conductivity!(scheme, mesh, T, ρ, cₚ, κ)
    # @timeit "check_diffusivity_validity" check_diffusivity_validity(scheme)
    @timeit "solve!" ImplicitSchemeType.solve!(
      scheme, mesh, T, dt; cutoff=cutoff, show_convergence=show_convergence, kwargs...
    )
  end
end

function nonlinear_thermal_conduction_step!(
  scheme::AbstractADESolver,
  mesh,
  T,
  ρ,
  cₚ,
  κ,
  Δt,
  nsubcycles=1;
  cutoff=false,
  show_convergence=true,
  apply_density_bc=true,
  kwargs...,
)
  for n in 1:nsubcycles
    dt = Δt / n
    # @timeit "applybc!" applybcs!(scheme.bcs, mesh, T)
    # display(T)

    # # if apply_density_bc
    # @timeit "applybc!" applybcs!(scheme.bcs, mesh, ρ)
    # display(ρ)
    # # end

    @timeit "update_conductivity!" update_conductivity!(scheme, mesh, T, ρ, cₚ, κ)

    @timeit "validate_diffusivity" validate_diffusivity(scheme)

    # @views begin
    #   @show extrema(scheme.α[scheme.iterators.domain.cartesian])
    # end

    @timeit "solve!" ADESolvers.solve!(scheme, mesh, T, dt; cutoff=cutoff, kwargs...)
  end
end
