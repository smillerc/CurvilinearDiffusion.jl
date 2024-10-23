
@kernel inbounds = true function _iter_param_kernel!(
  dτ_ρ, θr_dτ, _Vpdτ, L, _ρ, α, dt, β, I0
)
  idx = @index(Global, Cartesian)
  idx += I0

  _Re = π + sqrt(π^2 + (L^2 * _ρ[idx]) / (α[idx] * dt))
  _dτ_ρ = (_Vpdτ * L / (α[idx] * _Re)) * β
  _θr_dτ = (L / (_Vpdτ * _Re)) * β

  isvalid = (abs(α[idx]) > 0) && isfinite(α[idx])
  dτ_ρ[idx] = _dτ_ρ * isvalid
  θr_dτ[idx] = _θr_dτ * isvalid
  dτ_ρ[idx] = _dτ_ρ * isfinite(_dτ_ρ)
  θr_dτ[idx] = _θr_dτ * isfinite(_θr_dτ)

  # if !isfinite(dτ_ρ[idx]) || abs(dτ_ρ[idx]) > 1e20
  #   @show L _ρ[idx] α[idx] dt _Vpdτ
  #   @show _Re θr_dτ[idx] dτ_ρ[idx]
  #   @show _θr_dτ _dτ_ρ
  #   error("gah!")
  # end
end

function update_iteration_params!(
  solver::PseudoTransientSolver{N,T}, ρ, Vpdτ, Δt; iter_scale=1
) where {N,T}
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _iter_param_kernel!(solver.backend)(
    solver.dτ_ρ,
    solver.θr_dτ,
    Vpdτ,
    solver.L,
    ρ,
    solver.α,
    Δt,
    iter_scale,
    idx_offset;
    ndrange=size(domain),
  )

  KernelAbstractions.synchronize(solver.backend)
  return nothing
end
