module TimeStepControl

using UnPack
using LinearAlgebra
using CurvilinearGrids
using CartesianDomains

export max_dt, maxxx_dt

# function max_dt(uⁿ⁺¹::AbstractArray{T,2}, uⁿ::AbstractArray{T,2}, mesh, CFL, Δt) where {T}
#   Δtⁿ⁺¹ᵢ_numer = zero(T)
#   Δtⁿ⁺¹ⱼ_numer = zero(T)
#   Δtⁿ⁺¹ᵢ_denom = zero(T)
#   Δtⁿ⁺¹ⱼ_denom = zero(T)

#   ϵ = 5eps(T)
#   for idx in mesh.iterators.cell.domain
#     ᵢ₊₁ = shift(idx, 1, +1)
#     ᵢ₋₁ = shift(idx, 1, -1)
#     ⱼ₋₁ = shift(idx, 2, -1)
#     ⱼ₊₁ = shift(idx, 2, +1)

#     ∂u∂ξ = 0.5abs(uⁿ⁺¹[ᵢ₊₁] - uⁿ⁺¹[ᵢ₋₁])
#     ∂u∂η = 0.5abs(uⁿ⁺¹[ⱼ₊₁] - uⁿ⁺¹[ⱼ₋₁])
#     ∂u∂t = abs(uⁿ⁺¹[idx] - uⁿ[idx]) / Δt

#     vᵢ = ∂u∂t / ∂u∂ξ
#     vⱼ = ∂u∂t / ∂u∂η

#     V⃗ = ξt + v⃗ᵢ * ξ̃x + v⃗ⱼ * ξ̃y # contravariant velocity
#   end

#   return Δt_next
# end

# function max_dt(uⁿ⁺¹::AbstractArray{T,2}, uⁿ::AbstractArray{T,2}, mesh, CFL, Δt) where {T}
#   Δtⁿ⁺¹ᵢ_numer = zero(T)
#   Δtⁿ⁺¹ⱼ_numer = zero(T)
#   Δtⁿ⁺¹ᵢ_denom = zero(T)
#   Δtⁿ⁺¹ⱼ_denom = zero(T)

#   ϵ = 5eps(T)
#   for idx in mesh.iterators.cell.domain
#     ᵢ₊₁ = shift(idx, 1, +1)
#     ᵢ₋₁ = shift(idx, 1, -1)
#     ⱼ₋₁ = shift(idx, 2, -1)
#     ⱼ₊₁ = shift(idx, 2, +1)

#     ∂u∂ξ = 0.5abs(uⁿ⁺¹[ᵢ₊₁] - uⁿ⁺¹[ᵢ₋₁])
#     ∂u∂η = 0.5abs(uⁿ⁺¹[ⱼ₊₁] - uⁿ⁺¹[ⱼ₋₁])
#     ∂u∂t = abs(uⁿ⁺¹[idx] - uⁿ[idx]) / Δt

#     vᵢ = ∂u∂t / ∂u∂ξ
#     vⱼ = ∂u∂t / ∂u∂η

#     V⃗ = ξt + v⃗ᵢ * ξ̃x + v⃗ⱼ * ξ̃y # contravariant velocity
#   end

#   return Δt_next
# end

# function _max_dt(vx, vy, ξt, ξx, ξy)
#   ξnorm = sqrt(ξx^2 + ξy^2)

#   ξ̃x = ξx / ξnorm
#   ξ̃y = ξy / ξnorm

#   U = ξt + vx * ξ̃x + vy * ξ̃y # contravariant velocity

#   return inv(abs(U))
# end

function max_dt(uⁿ⁺¹, uⁿ, mesh, Δt)
  max_relative_Δu = -Inf

  @views begin
    umax = 0.5maximum(uⁿ[mesh.iterators.cell.domain])
  end

  ϵ = 0.001
  for idx in mesh.iterators.cell.domain
    max_relative_Δu = max(
      max_relative_Δu, #
      abs(uⁿ⁺¹[idx] - uⁿ[idx]) / (uⁿ⁺¹[idx] + ϵ * umax),
    )
  end

  η_target = 0.05 # 5% change

  Δtⁿ⁺¹ = Δt * sqrt(η_target / max_relative_Δu)

  dt_next = min(Δtⁿ⁺¹, 1.1Δt)

  return dt_next
end

# function max_dt(uⁿ⁺¹::AbstractArray{T,2}, uⁿ::AbstractArray{T,2}, mesh, CFL, Δt) where {T}
#   Δtⁿ⁺¹ᵢ_numer = zero(T)
#   Δtⁿ⁺¹ⱼ_numer = zero(T)
#   Δtⁿ⁺¹ᵢ_denom = zero(T)
#   Δtⁿ⁺¹ⱼ_denom = zero(T)

#   ϵ = 5eps(T)

#   dom = mesh.iterators.cell.domain
#   @views begin
#     L2 = norm(uⁿ⁺¹[dom] - uⁿ[dom])
#   end
#   @show L2

#   for idx in mesh.iterators.cell.domain
#     ᵢ₊₁ = shift(idx, 1, +1)
#     ᵢ₋₁ = shift(idx, 1, -1)
#     ⱼ₋₁ = shift(idx, 2, -1)
#     ⱼ₊₁ = shift(idx, 2, +1)

#     _Δtⁿ⁺¹ᵢ_numer = abs(uⁿ⁺¹[ᵢ₊₁] - uⁿ⁺¹[ᵢ₋₁])
#     _Δtⁿ⁺¹ⱼ_numer = abs(uⁿ⁺¹[ⱼ₊₁] - uⁿ⁺¹[ⱼ₋₁])

#     _Δtⁿ⁺¹ᵢ_denom = abs(uⁿ⁺¹[idx] - uⁿ[idx]) / Δt
#     _Δtⁿ⁺¹ⱼ_denom = abs(uⁿ⁺¹[idx] - uⁿ[idx]) / Δt

#     _Δtⁿ⁺¹ᵢ_numer = _Δtⁿ⁺¹ᵢ_numer * (abs(_Δtⁿ⁺¹ᵢ_numer) >= ϵ)
#     _Δtⁿ⁺¹ⱼ_numer = _Δtⁿ⁺¹ⱼ_numer * (abs(_Δtⁿ⁺¹ⱼ_numer) >= ϵ)
#     _Δtⁿ⁺¹ᵢ_denom = _Δtⁿ⁺¹ᵢ_denom * (abs(_Δtⁿ⁺¹ᵢ_denom) >= ϵ)
#     _Δtⁿ⁺¹ⱼ_denom = _Δtⁿ⁺¹ⱼ_denom * (abs(_Δtⁿ⁺¹ⱼ_denom) >= ϵ)

#     Δtⁿ⁺¹ᵢ_numer += _Δtⁿ⁺¹ᵢ_numer
#     Δtⁿ⁺¹ⱼ_numer += _Δtⁿ⁺¹ⱼ_numer
#     Δtⁿ⁺¹ᵢ_denom += _Δtⁿ⁺¹ᵢ_denom
#     Δtⁿ⁺¹ⱼ_denom += _Δtⁿ⁺¹ⱼ_denom
#   end

#   Δtⁿ⁺¹ᵢ = 0.5(Δtⁿ⁺¹ᵢ_numer / Δtⁿ⁺¹ᵢ_denom)
#   Δtⁿ⁺¹ⱼ = 0.5(Δtⁿ⁺¹ⱼ_numer / Δtⁿ⁺¹ⱼ_denom)

#   Δt_next = CFL * min(Δtⁿ⁺¹ᵢ, Δtⁿ⁺¹ⱼ)

#   @show Δtⁿ⁺¹ᵢ, Δtⁿ⁺¹ⱼ
#   return Δt_next
# end

end
