module TimeStepControl

using CurvilinearGrids: AbstractCurvilinearGrid

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

TBW
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
  max_relative_Δu = -Inf

  @views begin
    umax = 0.5maximum(uⁿ[mesh.iterators.cell.domain])
  end

  @inbounds for idx in mesh.iterators.cell.domain
    max_relative_Δu = max(
      max_relative_Δu, #
      abs(uⁿ⁺¹[idx] - uⁿ[idx]) / (uⁿ⁺¹[idx] + tol * umax),
    )
  end

  Δtⁿ⁺¹ = Δt * sqrt(maximum_u_change / max_relative_Δu)

  dt_next = min(Δtⁿ⁺¹, timestep_growth_factor * Δt)

  return dt_next
end

end
