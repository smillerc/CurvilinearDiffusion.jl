
function update_iteration_params!(
  solver::PseudoTransientSolver{N,T,BE}, ρ, Vpdτ, Δt; iter_scale=1
) where {N,T,BE<:CPU}

  #
  L = solver.L
  α = solver.α
  β = iter_scale
  dτ_ρ = solver.dτ_ρ
  θr_dτ = solver.θr_dτ

  @batch for idx in solver.iterators.domain.cartesian
    _Re = π + sqrt(π^2 + (L^2 * ρ[idx]) / (α[idx] * Δt))
    dτ_ρ[idx] = (Vpdτ * L / (α[idx] * _Re)) * β
    θr_dτ[idx] = (L / (Vpdτ * _Re)) * β
  end

  return nothing
end

function update_iteration_params!(
  solver::PseudoTransientSolver{N,T,BE}, ρ, Vpdτ, Δt; iter_scale=1
) where {N,T,BE<:GPU}

  #
  function f_dτ_ρ(Vpdτ, L, ρ, α, dt)
    Re = π + sqrt(π^2 + (L^2 * ρ) / (α * dt))
    return (Vpdτ * L / (α * Re))
  end

  function f_θr_dτ(Vpdτ, L, ρ, α, dt)
    Re = π + sqrt(π^2 + (L^2 * ρ) / (α * dt))
    return L / (Vpdτ * Re)
  end

  @. solver.dτ_ρ = f_dτ_ρ(Vpdτ, solver.L, ρ, solver.α, Δt)
  @. solver.θr_dτ = f_θr_dτ(Vpdτ, solver.L, ρ, solver.α, Δt)
end
