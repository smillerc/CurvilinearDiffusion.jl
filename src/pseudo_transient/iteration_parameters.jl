
function update_iteration_params!(
  solver::PseudoTransientSolver{N,T,BE}, ρ, Vpdτ, Δt; iter_scale=1
) where {N,T,BE<:CPU}

  #
  L = solver.L
  α = solver.α
  β = iter_scale
  Re = solver.Re
  dτ_ρ = solver.dτ_ρ
  θr_dτ = solver.θr_dτ

  @batch for idx in solver.iterators.domain.cartesian
    _Re = π + sqrt(π^2 + (L^2 * ρ[idx]) / (α[idx] * Δt))
    Re[idx] = _Re
    dτ_ρ[idx] = (Vpdτ * L / (α[idx] * _Re)) * β
    θr_dτ[idx] = (L / (Vpdτ * _Re)) * β
  end

  return nothing
end

function _update_iteration_params_gpu!(solver, ρ, Vpdτ, Δt; iter_scale=1)
  # β=1 / 1.2, # iteration scaling parameter
  # β = 1, # iteration scaling parameter
  #
  @kernel function _iter_param_kernel!(Re, dτ_ρ, θr_dτ, _Vpdτ, L, _ρ, α, dt, β, I0)
    idx = @index(Global, Cartesian)
    idx += I0

    @inbounds begin
      _Re = π + sqrt(π^2 + (L^2 * _ρ[idx]) / (α[idx] * dt))
      Re[idx] = _Re
      dτ_ρ[idx] = (_Vpdτ * L / (α[idx] * _Re)) * β
      θr_dτ[idx] = (L / (_Vpdτ * _Re)) * β
    end
  end

  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _iter_param_kernel!(solver.backend)(
    solver.Re,
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

# function update_iteration_params!(solver, ρ, Vpdτ, Δt; iter_scale=1)
# end