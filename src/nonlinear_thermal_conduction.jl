using TimerOutputs

function nonlinear_thermal_conduction_step!(
  scheme::ImplicitScheme, mesh, T, ρ, cₚ, κ, Δt; cutoff=true
)
  @timeit "applybc!" applybcs!(scheme.bcs, mesh, T)
  # @timeit "applybc!" applybcs!(scheme.bcs, mesh, ρ)
  @timeit "update_conductivity!" update_conductivity!(scheme, T, ρ, cₚ, κ)
  @timeit "check_diffusivity_validity" check_diffusivity_validity(scheme)
  @timeit "solve!" solve!(scheme, mesh, T, Δt; cutoff=cutoff)
end
