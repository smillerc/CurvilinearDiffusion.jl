using KernelAbstractions

"""
When using the diffusion solver to solve the heat equation, this can be used to
update the thermal conductivity of the mesh at each cell-center.

# Arguments
 - `diffusivity::AbstractArray`: diffusivity of the problem
 - `temp::AbstractArray`: temperature
 - `ρ::AbstractArray`: density
 - `κ::Function`: function to determine thermal conductivity, e.g. κ(ρ,T) = κ0 * ρ * T^(5/2)
 - `cₚ::Real`: heat capacity at constant pressure
"""
function update_conductivity!(diffusivity, temp, ρ, cₚ, κ::F) where {F<:Function}
  backend = KernelAbstractions.get_backend(diffusivity)
  conductivity_kernel(backend)(diffusivity, temp, ρ, cₚ, κ; ndrange=size(diffusivity))

  return nothing
end

# conductivity with array-valued cₚ
@kernel function conductivity_kernel(
  α, temperature, density, cₚ::AbstractArray{T,N}, κ::F
) where {T,N,F<:Function}
  idx = @index(Global)

  @inbounds begin
    α[idx] = κ(density[idx], temperature[idx]) / (density[idx] * cₚ[idx])
  end
end

# conductivity with single-valued cₚ
@kernel function conductivity_kernel(
  α, temperature, density, cₚ::Real, κ::F
) where {F<:Function}
  idx = @index(Global)

  @inbounds begin
    α[idx] = κ(density[idx], temperature[idx]) / (density[idx] * cₚ)
  end
end
