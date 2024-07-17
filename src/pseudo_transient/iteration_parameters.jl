function update_iteration_params!(solver, ρ, Vpdτ, Δt; iter_scale=1)
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