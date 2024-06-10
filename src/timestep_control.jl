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
  mesh::AbstractCurvilinearGrid,
  Δt;
  u0=max,
  tol=1e-3,
  maximum_u_change=0.05,
  timestep_growth_factor=1.1,
)
  domain = mesh.iterators.cell.domain
  @views begin
    umax = 0.5maximum(uⁿ[domain])
  end

  backend = get_backend(uⁿ)
  u_0 = umax * tol
  max_relative_Δu = _max_relative_change(uⁿ⁺¹, uⁿ, domain, u_0, backend)

  Δtⁿ⁺¹ = Δt * sqrt(maximum_u_change / max_relative_Δu)

  dt_next = min(Δtⁿ⁺¹, timestep_growth_factor * Δt)

  return dt_next
end

@inline function _max_relative_change(uⁿ⁺¹, uⁿ, domain, u0, ::CPU)
  max_relative_Δu = -Inf

  @inbounds for idx in domain
    max_relative_Δu = max(max_relative_Δu, abs(uⁿ⁺¹[idx] - uⁿ[idx]) / (uⁿ⁺¹[idx] + u0))
  end

  return max_relative_Δu
end

@inline function _max_relative_change(uⁿ⁺¹, uⁿ, domain, u0, ::GPU)
  f(a, b) = abs(a - b) / (a + u0)

  @views begin
    max_relative_Δu = mapreduce(f, max, uⁿ⁺¹[domain], uⁿ[domain])
  end

  return max_relative_Δu
end

end
