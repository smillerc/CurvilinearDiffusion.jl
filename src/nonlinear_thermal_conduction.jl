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
)
  for n in 1:nsubcycles
    dt = Δt / n
    @timeit "applybc!" applybcs!(scheme.bcs, mesh, T)
    @timeit "applybc!" applybcs!(scheme.bcs, mesh, ρ)
    @timeit "update_conductivity!" update_conductivity!(scheme, T, ρ, cₚ, κ)
    @timeit "check_diffusivity_validity" check_diffusivity_validity(scheme)
    @timeit "solve!" solve!(
      scheme, mesh, T, dt; cutoff=cutoff, show_convergence=show_convergence
    )
  end
end
