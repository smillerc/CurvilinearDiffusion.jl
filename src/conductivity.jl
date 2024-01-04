"""
When using the diffusion solver to solve the heat equation, this can be used to
update the thermal conductivity of the mesh at each cell-center.

# Arguments
 - `solver::ADESolver`: The solver type
 - `T::Array`: Temperature
 - `ρ::Array`: Density
 - `κ::Function`: Function to determine thermal conductivity, e.g. κ(ρ,T) = κ0 * ρ * T^(5/2)
 - `cₚ::Real`: Heat capacity
"""
function update_conductivity!(solver, T::Array, ρ::Array, κ::Function, cₚ::Real)
  @batch for idx in eachindex(T)
    rho = ρ[idx]
    kappa = κ(rho, T[idx])
    solver.α[idx] = kappa / (rho * cₚ)
  end

  return nothing
end
