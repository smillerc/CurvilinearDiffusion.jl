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
function update_conductivity!(
  diffusivity::AbstractArray{T,N},
  temp::AbstractArray{T,N},
  ρ::AbstractArray{T,N},
  κ::F,
  cₚ::Real,
) where {T,N,F}
  backend = KernelAbstractions.get_backend(diffusivity)
  conductivity_kernel(backend)(diffusivity, temp, ρ, κ, cₚ; ndrange=size(temp))

  return nothing
end

# super-simple kernel to update the diffusivity if
# we know the conductivity function
@kernel function conductivity_kernel(
  α, @Const(temperature), @Const(density), @Const(κ::F), @Const(cₚ)
) where {F}
  idx = @index(Global)

  @inbounds begin
    rho = density[idx]
    α[idx] = κ(rho, temperature[idx]) / (rho * cₚ)
  end
end
