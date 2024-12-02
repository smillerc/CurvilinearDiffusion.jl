
@kernel inbounds = true function _iter_param_kernel!(dτ_ρ, θr_dτ, _Vpdτ, L, _ρ, α, dt, s)
  idx = @index(Global, Cartesian)

  _Re = π + sqrt(π^2 + (L^2 * _ρ[idx]) / (α[idx] * dt))
  _dτ_ρ = (_Vpdτ * L / (α[idx] * _Re)) # * β
  _θr_dτ = (L / (_Vpdτ * _Re))

  # add in source term dependency somewhere in here?? Maybe this could help?
  isvalid = (abs(α[idx]) > 0) && isfinite(α[idx])
  dτ_ρ[idx] = _dτ_ρ * isvalid
  θr_dτ[idx] = _θr_dτ * isvalid
  dτ_ρ[idx] = _dτ_ρ * isfinite(_dτ_ρ)
  θr_dτ[idx] = _θr_dτ * isfinite(_θr_dτ)
end

function update_iteration_params!(
  solver::PseudoTransientSolver{N,T}, ρ, Vpdτ, Δt; iter_scale=sqrt(2)
) where {N,T}
  _iter_param_kernel!(solver.backend)(
    solver.dτ_ρ,
    solver.θr_dτ,
    Vpdτ,
    solver.L,
    ρ,
    solver.α,
    Δt,
    solver.source_term;
    ndrange=size(solver.dτ_ρ),
  )

  # KernelAbstractions.synchronize(solver.backend)
  return nothing
end
