using TimerOutputs

function nonlinear_thermal_conduction_step!(scheme::ImplicitScheme, mesh, T, ρ, cₚ, κ, Δt)
  @timeit "applybc!" applybcs!(scheme.bcs, mesh, T)
  # @timeit "applybc!" applybcs!(scheme.bcs, mesh, ρ)
  @timeit "update_conductivity!" update_conductivity!(scheme, T, ρ, cₚ, κ)
  @timeit "solve!" solve!(scheme, mesh, T, Δt)
end
