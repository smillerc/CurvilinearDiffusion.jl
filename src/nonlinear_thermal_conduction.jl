using TimerOutputs

function nonlinear_thermal_conduction_step!(scheme::ImplicitScheme, mesh, T, ρ, cₚ, κ, Δt)
  @timeit "applybc!" applybcs!(scheme.bcs, mesh, scheme.limits, T)
  @timeit "applybc!" applybcs!(scheme.bcs, mesh, scheme.limits, ρ)
  @timeit "update_conductivity!" update_conductivity!(scheme.α, T, ρ, cₚ, κ)
  @timeit "solve!" solve!(scheme, mesh, T, Δt)
end
