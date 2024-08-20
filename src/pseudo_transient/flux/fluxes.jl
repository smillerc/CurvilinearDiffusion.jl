
include("fluxes_gpu.jl")
include("fluxes_cpu.jl")

function flux_kernel!(
  qᵢ₊½::T, uᵢ₊₁, uᵢ, αᵢ₊₁, αᵢ, θr_dτᵢ₊₁, θr_dτᵢ, mean::F, rtol=sqrt(eps(eltype(T)))
) where {T,F}
  αᵢ₊½ = mean(αᵢ, αᵢ₊₁)
  θr_dτ_ᵢ₊½ = mean(θr_dτᵢ, θr_dτᵢ₊₁)

  du = uᵢ₊₁ - uᵢ
  du = du * (abs(uᵢ * rtol) < abs(du)) # perform epsilon check
  _qᵢ₊½ = -αᵢ₊½ * du

  qᵢ₊½ = (qᵢ₊½ * θr_dτ_ᵢ₊½ + _qᵢ₊½) / (1 + θr_dτ_ᵢ₊½)
  return qᵢ₊½
end

function fluxprime_kernel!(
  uᵢ₊₁::T, uᵢ, αᵢ₊₁, αᵢ, mean::F, rtol=sqrt(eps(eltype(T)))
) where {T,F}

  #
  αᵢ₊½ = mean(αᵢ, αᵢ₊₁)

  du = uᵢ₊₁ - uᵢ
  du = du * (abs(uᵢ * rtol) < abs(du)) # perform epsilon check
  return -αᵢ₊½ * du
end
