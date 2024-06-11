module TimeStepControl

using CurvilinearGrids: AbstractCurvilinearGrid
using KernelAbstractions: GPU, CPU, get_backend

export next_dt

"""
    next_dt(
  uⁿ⁺¹,
  uⁿ,
  mesh,
  Δt;
  u0=max,
  tol=1e-3,
  maximum_u_change=0.05,
  timestep_growth_factor=1.1,
)


# Arguments
 - `uⁿ⁺¹`
 - `uⁿ`
 - `mesh`
 - `Δt`

# Keyword Arguments
 - `u0`=max: 
 - `tol`=1e-3: 
 - `maximum_u_change`=0.05: 
 - `timestep_growth_factor`=1.1: 
"""
function next_dt(
  uⁿ⁺¹,
  uⁿ,
  Δt;
  u0=max,
  tol=1e-3,
  maximum_u_change=0.05,
  timestep_growth_factor=1.1,
  kwargs...,
)
  umax = maximum(uⁿ)

  backend = get_backend(uⁿ)
  u_0 = umax * tol
  max_relative_Δu = _max_relative_change(uⁿ⁺¹, uⁿ, u_0, backend)

  Δtⁿ⁺¹ = Δt * sqrt(maximum_u_change / max_relative_Δu)

  dt_next = min(Δtⁿ⁺¹, timestep_growth_factor * Δt)

  return dt_next
end

@inline function _max_relative_change(uⁿ⁺¹, uⁿ, u0, ::CPU)
  max_relative_Δu = -Inf

  for (i1, i2) in zip(eachindex(uⁿ), eachindex(uⁿ⁺¹))
    δ = abs(uⁿ⁺¹[i2] - uⁿ[i1]) / (uⁿ⁺¹[i2] + u0)
    max_relative_Δu = max(max_relative_Δu, δ * isfinite(δ))
  end

  return max_relative_Δu
end

@inline function _max_relative_change(uⁿ⁺¹, uⁿ, u0, ::GPU)
  function f(a, b)
    δ = abs(a - b) / (a + u0)
    return δ * isfinite(δ)
  end
  return mapreduce(f, max, uⁿ⁺¹, uⁿ)
end

end
