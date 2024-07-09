module PseudoTransientScheme

using LinearAlgebra: norm

using TimerOutputs: @timeit
using CartesianDomains: expand, shift, expand_lower, haloedge_regions
using CurvilinearGrids: CurvilinearGrid2D, CurvilinearGrid3D, cellsize_withhalo, coords
using KernelAbstractions
using KernelAbstractions: CPU, GPU, @kernel, @index
using Polyester: @batch
using UnPack: @unpack

using ..BoundaryConditions

include("../averaging.jl")
include("../validity_checks.jl")
include("../edge_terms.jl")

export PseudoTransientSolver

struct PseudoTransientSolver{N,T,BE,AA<:AbstractArray{T,N},NT1,DM,B,F}
  u::AA
  u_prev::AA
  source_term::AA
  q::NT1
  q′::NT1
  residual::AA
  Reynolds_number::AA
  α::AA # diffusivity
  θr_dτ::AA
  dτ_ρ::AA
  spacing::NTuple{N,T}
  L::T
  iterators::DM
  bcs::B # boundary conditions
  mean::F
  backend::BE
end

function PseudoTransientSolver(
  mesh::CurvilinearGrid2D, bcs; backend=CPU(), face_diffusivity=:arithmetic, T=Float64
)
  #
  #         u
  #     |--------|--------|--------|--------|
  #           q_i+1/2
  #

  iterators = (
    domain=(cartesian=mesh.iterators.cell.domain,),
    full=(cartesian=mesh.iterators.cell.full,),
  )

  # cell-based
  u = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  u_prev = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  S = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)) # source term

  # edge-based
  q = (
    x=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    y=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
  )
  q′ = (
    x=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    y=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
  )

  residual = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  Reynolds_number = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))

  α = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))

  θr_dτ = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  dτ_ρ = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))

  x, y = coords(mesh)
  spacing = (minimum(diff(x; dims=1)), minimum(diff(y; dims=2))) .|> T
  min_x, max_x = extrema(x)
  min_y, max_y = extrema(y)
  L = max(abs(max_x - min_x), abs(max_y - min_y)) |> T

  if face_diffusivity === :harmonic
    mean_func = harmonic_mean # from ../averaging.jl
  else
    mean_func = arithmetic_mean # from ../averaging.jl
  end

  return PseudoTransientSolver(
    u,
    u_prev,
    S,
    q,
    q′,
    residual,
    Reynolds_number,
    α,
    θr_dτ,
    dτ_ρ,
    spacing,
    L,
    iterators,
    bcs,
    mean_func,
    backend,
  )
end

function PseudoTransientSolver(
  mesh::CurvilinearGrid3D, bcs; backend=CPU(), face_diffusivity=:arithmetic, T=Float64
)
  #
  #         u
  #     |--------|--------|--------|--------|
  #           q_i+1/2
  #

  iterators = (
    domain=(cartesian=mesh.iterators.cell.domain,),
    full=(cartesian=mesh.iterators.cell.full,),
  )

  # cell-based
  u = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  u_prev = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  S = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)) # source term

  # edge-based
  q = (
    x=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    y=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    z=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
  )
  q′ = (
    x=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    y=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    z=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
  )

  residual = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  Reynolds_number = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))

  α = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))

  θr_dτ = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  dτ_ρ = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))

  x, y, z = coords(mesh)
  spacing =
    (minimum(diff(x; dims=1)), minimum(diff(y; dims=2)), minimum(diff(z; dims=3))) .|> T

  min_x, max_x = extrema(x)
  min_y, max_y = extrema(y)
  min_z, max_z = extrema(z)

  L = max(abs(max_x - min_x), abs(max_y - min_y), abs(max_z - min_z)) |> T

  if face_diffusivity === :harmonic
    mean_func = harmonic_mean # from ../averaging.jl
  else
    mean_func = arithmetic_mean # from ../averaging.jl
  end

  return PseudoTransientSolver(
    u,
    u_prev,
    S,
    q,
    q′,
    residual,
    Reynolds_number,
    α,
    θr_dτ,
    dτ_ρ,
    spacing,
    L,
    iterators,
    bcs,
    mean_func,
    backend,
  )
end

# solve a single time-step dt
function step!(
  solver::PseudoTransientSolver{N},
  mesh,
  T,
  ρ,
  cₚ,
  κ,
  dt;
  max_iter=1e5,
  tol=1e-8,
  error_check_interval=10,
  cutoff=true,
) where {N}

  #
  domain = solver.iterators.domain.cartesian
  nhalo = 1

  iter = 0
  err = 2 * tol

  CFL = 1 / sqrt(N)

  dx, dy = solver.spacing
  Vpdτ = CFL * min(dx, dy)

  copy!(solver.u, T)
  copy!(solver.u_prev, solver.u)

  update_conductivity!(solver, mesh, solver.u, ρ, cₚ, κ)

  validate_scalar(solver.u, domain, nhalo, :u; enforce_positivity=true)
  validate_scalar(solver.source_term, domain, nhalo, :source_term; enforce_positivity=false)
  validate_scalar(solver.α, domain, nhalo, :diffusivity; enforce_positivity=true)

  # Pseudo-transient iteration
  while err > tol && iter < max_iter && isfinite(err)
    # @info "Iter: $iter, Err: $err"
    iter += 1

    # Diffusion coefficient
    if iter > 1
      update_conductivity!(solver, mesh, solver.u, ρ, cₚ, κ)
    end
    applybcs!(solver.bcs, mesh, solver.u)

    @timeit "update_iteration_params!" update_iteration_params!(solver, ρ, Vpdτ, dt;)

    applybcs!(solver.bcs, mesh, solver.α)
    applybcs!(solver.bcs, mesh, solver.dτ_ρ)
    applybcs!(solver.bcs, mesh, solver.θr_dτ)

    @timeit "compute_flux!" compute_flux!(solver, mesh)
    @timeit "compute_update!" compute_update!(solver, mesh, dt)

    # Apply a cutoff function to remove negative
    if cutoff
      cutoff!(solver.u)
    end

    if iter % error_check_interval == 0
      @timeit "update_residual!" update_residual!(solver, mesh, dt)

      @timeit "norm" begin
        residual = @view solver.residual[solver.iterators.domain.cartesian]
        err = L2_norm(residual)
      end
      if !isfinite(err)
        @show extrema(solver.residual)
        @error("Non-finite error detected! ($err)")
      end
    end
  end

  if iter == max_iter
    error(
      "Maximum iteration limit reached ($max_iter), current error is $(err), tolerance is $tol, exiting...",
    )
  end

  validate_scalar(solver.u, domain, nhalo, :u; enforce_positivity=true)
  copy!(solver.u_prev, solver.u)
  copy!(T, solver.u)

  return err, iter
end

function L2_norm(A)
  _norm = sqrt(mapreduce(x -> x^2, +, A)) / sqrt(length(A))
  return _norm
end

function update_iteration_params!(solver, ρ, Vpdτ, Δt; iter_scale=1)
  @kernel function _iter_param_kernel!(
    Reynolds_number, dτ_ρ, θr_dτ, _Vpdτ, L, _ρ, α, dt, β, index_offset
  )
    idx = @index(Global, Cartesian)
    idx += index_offset

    @inbounds begin
      _Re = π + sqrt(π^2 + (L^2 * _ρ[idx]) / (α[idx] * dt))
      Reynolds_number[idx] = _Re
      dτ_ρ[idx] = (_Vpdτ * L / (α[idx] * _Re)) * β
      θr_dτ[idx] = (L / (_Vpdτ * _Re)) * β
    end
  end

  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _iter_param_kernel!(solver.backend)(
    solver.Reynolds_number,
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

@kernel function flux_kernel!(
  qᵢ₊½, q′ᵢ₊½, u, α, θr_dτ, axis, index_offset, mean_func::F
) where {F}
  idx = @index(Global, Cartesian)
  ᵢ = idx + index_offset

  @inbounds begin
    ᵢ₊₁ = shift(ᵢ, axis, +1)

    # edge diffusivity / iter params
    αᵢ₊½ = mean_func(α[ᵢ], α[ᵢ₊₁])
    θr_dτ_ᵢ₊½ = (θr_dτ[ᵢ] + θr_dτ[ᵢ₊₁]) / 2

    _qᵢ₊½ = -αᵢ₊½ * (u[ᵢ₊₁] - u[ᵢ])

    qᵢ₊½[ᵢ] = (qᵢ₊½[ᵢ] * θr_dτ_ᵢ₊½ + _qᵢ₊½) / (1 + θr_dτ_ᵢ₊½)
    q′ᵢ₊½[ᵢ] = _qᵢ₊½
  end
end

function compute_flux!(solver::PseudoTransientSolver{2,T}, ::CurvilinearGrid2D) where {T}
  iaxis = 1
  jaxis = 2
  ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)

  ᵢ₊½_idx_offset = first(ᵢ₊½_domain) - oneunit(first(ᵢ₊½_domain))
  ⱼ₊½_idx_offset = first(ⱼ₊½_domain) - oneunit(first(ⱼ₊½_domain))

  flux_kernel!(solver.backend)(
    solver.q.x,
    solver.q′.x,
    solver.u,
    solver.α,
    solver.θr_dτ,
    iaxis,
    ᵢ₊½_idx_offset,
    solver.mean;
    ndrange=size(ᵢ₊½_domain),
  )

  flux_kernel!(solver.backend)(
    solver.q.y,
    solver.q′.y,
    solver.u,
    solver.α,
    solver.θr_dτ,
    jaxis,
    ⱼ₊½_idx_offset,
    solver.mean;
    ndrange=size(ⱼ₊½_domain),
  )

  KernelAbstractions.synchronize(solver.backend)

  return nothing
end

function compute_flux!(solver::PseudoTransientSolver{3,T}, ::CurvilinearGrid3D) where {T}
  iaxis = 1
  jaxis = 2
  kaxis = 3

  ᵢ₊½_domain = expand_lower(solver.iterators.domain.cartesian, iaxis, +1)
  ⱼ₊½_domain = expand_lower(solver.iterators.domain.cartesian, jaxis, +1)
  ₖ₊½_domain = expand_lower(solver.iterators.domain.cartesian, kaxis, +1)

  ᵢ₊½_idx_offset = first(ᵢ₊½_domain) - oneunit(first(ᵢ₊½_domain))
  ⱼ₊½_idx_offset = first(ⱼ₊½_domain) - oneunit(first(ⱼ₊½_domain))
  ₖ₊½_idx_offset = first(ₖ₊½_domain) - oneunit(first(ₖ₊½_domain))

  flux_kernel!(solver.backend)(
    solver.q.x,
    solver.q′.x,
    solver.u,
    solver.α,
    solver.θr_dτ,
    iaxis,
    ᵢ₊½_idx_offset,
    solver.mean;
    ndrange=size(ᵢ₊½_domain),
  )

  flux_kernel!(solver.backend)(
    solver.q.y,
    solver.q′.y,
    solver.u,
    solver.α,
    solver.θr_dτ,
    jaxis,
    ⱼ₊½_idx_offset,
    solver.mean;
    ndrange=size(ⱼ₊½_domain),
  )

  flux_kernel!(solver.backend)(
    solver.q.z,
    solver.q′.z,
    solver.u,
    solver.α,
    solver.θr_dτ,
    kaxis,
    ₖ₊½_idx_offset,
    solver.mean;
    ndrange=size(ₖ₊½_domain),
  )

  KernelAbstractions.synchronize(solver.backend)

  return nothing
end

@kernel function _update_kernel!(
  u,
  u_prev,
  cell_center_metrics, # applying @Const to a struct array causes problems
  edge_metrics, # applying @Const to a struct array causes problems
  α,
  q,
  dτ_ρ,
  source_term,
  dt,
  index_offset,
)
  idx = @index(Global, Cartesian)
  idx += index_offset

  @inbounds begin
    @inline ∇q = flux_divergence(q, u, α, cell_center_metrics, edge_metrics, idx)

    u[idx] = (
      (u[idx] + dτ_ρ[idx] * (u_prev[idx] / dt - ∇q + source_term[idx])) /
      (1 + dτ_ρ[idx] / dt)
    )
  end
end

function compute_update!(solver, mesh, Δt)
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _update_kernel!(solver.backend)(
    solver.u,
    solver.u_prev,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    solver.α,
    solver.q,
    solver.dτ_ρ,
    solver.source_term,
    Δt,
    idx_offset;
    ndrange=size(domain),
  )

  KernelAbstractions.synchronize(solver.backend)
  return nothing
end

function flux_divergence(
  (qᵢ, qⱼ), u, α, cell_center_metrics, edge_metrics, idx::CartesianIndex{2}
)
  i, j = idx.I

  Jᵢ₊½ = edge_metrics.i₊½.J[i, j]
  Jⱼ₊½ = edge_metrics.j₊½.J[i, j]
  Jᵢ₋½ = edge_metrics.i₊½.J[i - 1, j]
  Jⱼ₋½ = edge_metrics.j₊½.J[i, j - 1]

  ξx = cell_center_metrics.ξ.x₁[i, j]
  ξy = cell_center_metrics.ξ.x₂[i, j]
  ηx = cell_center_metrics.η.x₁[i, j]
  ηy = cell_center_metrics.η.x₂[i, j]

  ξxᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[i, j] / Jᵢ₊½
  ξyᵢ₊½ = edge_metrics.i₊½.ξ̂.x₂[i, j] / Jᵢ₊½
  ηxᵢ₊½ = edge_metrics.i₊½.η̂.x₁[i, j] / Jᵢ₊½
  ηyᵢ₊½ = edge_metrics.i₊½.η̂.x₂[i, j] / Jᵢ₊½

  ξxᵢ₋½ = edge_metrics.i₊½.ξ̂.x₁[i - 1, j] / Jᵢ₋½
  ξyᵢ₋½ = edge_metrics.i₊½.ξ̂.x₂[i - 1, j] / Jᵢ₋½
  ηxᵢ₋½ = edge_metrics.i₊½.η̂.x₁[i - 1, j] / Jᵢ₋½
  ηyᵢ₋½ = edge_metrics.i₊½.η̂.x₂[i - 1, j] / Jᵢ₋½

  ξxⱼ₊½ = edge_metrics.j₊½.ξ̂.x₁[i, j] / Jⱼ₊½
  ξyⱼ₊½ = edge_metrics.j₊½.ξ̂.x₂[i, j] / Jⱼ₊½
  ηxⱼ₊½ = edge_metrics.j₊½.η̂.x₁[i, j] / Jⱼ₊½
  ηyⱼ₊½ = edge_metrics.j₊½.η̂.x₂[i, j] / Jⱼ₊½

  ξxⱼ₋½ = edge_metrics.j₊½.ξ̂.x₁[i, j - 1] / Jⱼ₋½
  ξyⱼ₋½ = edge_metrics.j₊½.ξ̂.x₂[i, j - 1] / Jⱼ₋½
  ηxⱼ₋½ = edge_metrics.j₊½.η̂.x₁[i, j - 1] / Jⱼ₋½
  ηyⱼ₋½ = edge_metrics.j₊½.η̂.x₂[i, j - 1] / Jⱼ₋½

  # flux divergence

  aᵢⱼ = (
    ξx * (ξxᵢ₊½ - ξxᵢ₋½) +
    ξy * (ξyᵢ₊½ - ξyᵢ₋½) +
    ηx * (ξxⱼ₊½ - ξxⱼ₋½) +
    ηy * (ξyⱼ₊½ - ξyⱼ₋½)
  )

  bᵢⱼ = (
    ξx * (ηxᵢ₊½ - ηxᵢ₋½) +
    ξy * (ηyᵢ₊½ - ηyᵢ₋½) +
    ηx * (ηxⱼ₊½ - ηxⱼ₋½) +
    ηy * (ηyⱼ₊½ - ηyⱼ₋½)
  )

  ∂qᵢ∂ξ = (ξx^2 + ξy^2) * (qᵢ[i, j] - qᵢ[i - 1, j])
  ∂qⱼ∂η = (ηx^2 + ηy^2) * (qⱼ[i, j] - qⱼ[i, j - 1])

  # ∂/∂η (α ∂u/∂ξ) aka ∂qᵢ/∂η
  ∂qᵢ∂η =
    0.5(ηx * ξx + ηy * ξy) * (
      # take average and do diff in j (for ∂/∂η)
      0.5(qᵢ[i, j + 1] + qᵢ[i - 1, j + 1]) - #
      0.5(qᵢ[i, j - 1] + qᵢ[i - 1, j - 1])   # 
    )

  # ∂/∂ξ (α ∂u/∂η)
  ∂qⱼ∂ξ =
    0.5(ηx * ξx + ηy * ξy) * (
      #  take average and do diff in i (for ∂/∂ξ)
      0.5(qⱼ[i + 1, j] + qⱼ[i + 1, j - 1]) - #
      0.5(qⱼ[i - 1, j] + qⱼ[i - 1, j - 1])   # 
    )

  ∂q∂ξ_nonorth = aᵢⱼ * 0.5(qᵢ[i, j] + qᵢ[i - 1, j]) # non-orth terms
  ∂q∂η_nonorth = bᵢⱼ * 0.5(qⱼ[i, j] + qⱼ[i, j - 1]) # non-orth terms

  ∇q = ∂qᵢ∂ξ + ∂qⱼ∂η + ∂qᵢ∂η + ∂qⱼ∂ξ + ∂q∂ξ_nonorth + ∂q∂η_nonorth
  return ∇q
end

function flux_divergence(
  (qᵢ, qⱼ, qₖ), u, α, cell_center_metrics, edge_metrics, idx::CartesianIndex{3}
)
  i, j, k = idx.I

  idim, jdim, kdim = (1, 2, 3)
  ᵢ₋₁ = shift(idx, idim, -1)
  ⱼ₋₁ = shift(idx, jdim, -1)
  ₖ₋₁ = shift(idx, kdim, -1)

  Jᵢ₊½ = edge_metrics.i₊½.J[idx]
  Jⱼ₊½ = edge_metrics.j₊½.J[idx]
  Jₖ₊½ = edge_metrics.k₊½.J[idx]

  Jᵢ₋½ = edge_metrics.i₊½.J[ᵢ₋₁]
  Jⱼ₋½ = edge_metrics.j₊½.J[ⱼ₋₁]
  Jₖ₋½ = edge_metrics.k₊½.J[ₖ₋₁]

  ξx = cell_center_metrics.ξ.x₁[idx]
  ξy = cell_center_metrics.ξ.x₂[idx]
  ξz = cell_center_metrics.ξ.x₃[idx]

  ηx = cell_center_metrics.η.x₁[idx]
  ηy = cell_center_metrics.η.x₂[idx]
  ηz = cell_center_metrics.η.x₃[idx]

  ζx = cell_center_metrics.ζ.x₁[idx]
  ζy = cell_center_metrics.ζ.x₂[idx]
  ζz = cell_center_metrics.ζ.x₃[idx]

  # ξxᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[idx] / Jᵢ₊½
  # ξyᵢ₊½ = edge_metrics.i₊½.ξ̂.x₂[idx] / Jᵢ₊½
  # ξzᵢ₊½ = edge_metrics.i₊½.ξ̂.x₃[idx] / Jᵢ₊½

  # ξxᵢ₋½ = edge_metrics.i₊½.ξ̂.x₁[ᵢ₋₁] / Jᵢ₋½
  # ξyᵢ₋½ = edge_metrics.i₊½.ξ̂.x₂[ᵢ₋₁] / Jᵢ₋½
  # ξzᵢ₋½ = edge_metrics.i₊½.ξ̂.x₃[ᵢ₋₁] / Jᵢ₋½

  # ηxᵢ₊½ = edge_metrics.i₊½.η̂.x₁[idx] / Jᵢ₊½
  # ηyᵢ₊½ = edge_metrics.i₊½.η̂.x₂[idx] / Jᵢ₊½
  # ηzᵢ₊½ = edge_metrics.i₊½.η̂.x₃[idx] / Jᵢ₊½

  # ηxᵢ₋½ = edge_metrics.i₊½.η̂.x₁[ᵢ₋₁] / Jᵢ₋½
  # ηyᵢ₋½ = edge_metrics.i₊½.η̂.x₂[ᵢ₋₁] / Jᵢ₋½
  # ηzᵢ₋½ = edge_metrics.i₊½.η̂.x₃[ᵢ₋₁] / Jᵢ₋½

  # ζxᵢ₊½ = edge_metrics.i₊½.ζ̂.x₁[idx] / Jᵢ₊½
  # ζyᵢ₊½ = edge_metrics.i₊½.ζ̂.x₂[idx] / Jᵢ₊½
  # ζzᵢ₊½ = edge_metrics.i₊½.ζ̂.x₃[idx] / Jᵢ₊½

  # ζxᵢ₋½ = edge_metrics.i₊½.ζ̂.x₁[ᵢ₋₁] / Jᵢ₋½
  # ζyᵢ₋½ = edge_metrics.i₊½.ζ̂.x₂[ᵢ₋₁] / Jᵢ₋½
  # ζzᵢ₋½ = edge_metrics.i₊½.ζ̂.x₃[ᵢ₋₁] / Jᵢ₋½

  # ξxⱼ₊½ = edge_metrics.j₊½.ξ̂.x₁[idx] / Jⱼ₊½
  # ξyⱼ₊½ = edge_metrics.j₊½.ξ̂.x₂[idx] / Jⱼ₊½
  # ξzⱼ₊½ = edge_metrics.j₊½.ξ̂.x₃[idx] / Jⱼ₊½

  # ξxⱼ₋½ = edge_metrics.j₊½.ξ̂.x₁[ⱼ₋₁] / Jⱼ₋½
  # ξyⱼ₋½ = edge_metrics.j₊½.ξ̂.x₂[ⱼ₋₁] / Jⱼ₋½
  # ξzⱼ₋½ = edge_metrics.j₊½.ξ̂.x₃[ⱼ₋₁] / Jⱼ₋½

  # ηxⱼ₊½ = edge_metrics.j₊½.η̂.x₁[idx] / Jⱼ₊½
  # ηyⱼ₊½ = edge_metrics.j₊½.η̂.x₂[idx] / Jⱼ₊½
  # ηzⱼ₊½ = edge_metrics.j₊½.η̂.x₃[idx] / Jⱼ₊½

  # ηxⱼ₋½ = edge_metrics.j₊½.η̂.x₁[ⱼ₋₁] / Jⱼ₋½
  # ηyⱼ₋½ = edge_metrics.j₊½.η̂.x₂[ⱼ₋₁] / Jⱼ₋½
  # ηzⱼ₋½ = edge_metrics.j₊½.η̂.x₃[ⱼ₋₁] / Jⱼ₋½

  # ζxⱼ₊½ = edge_metrics.j₊½.ζ̂.x₁[idx] / Jⱼ₊½
  # ζyⱼ₊½ = edge_metrics.j₊½.ζ̂.x₂[idx] / Jⱼ₊½
  # ζzⱼ₊½ = edge_metrics.j₊½.ζ̂.x₃[idx] / Jⱼ₊½

  # ζxⱼ₋½ = edge_metrics.j₊½.ζ̂.x₁[ⱼ₋₁] / Jⱼ₋½
  # ζyⱼ₋½ = edge_metrics.j₊½.ζ̂.x₂[ⱼ₋₁] / Jⱼ₋½
  # ζzⱼ₋½ = edge_metrics.j₊½.ζ̂.x₃[ⱼ₋₁] / Jⱼ₋½

  # ξxₖ₊½ = edge_metrics.k₊½.ξ̂.x₁[idx] / Jₖ₊½
  # ξyₖ₊½ = edge_metrics.k₊½.ξ̂.x₂[idx] / Jₖ₊½
  # ξzₖ₊½ = edge_metrics.k₊½.ξ̂.x₃[idx] / Jₖ₊½

  # ξxₖ₋½ = edge_metrics.k₊½.ξ̂.x₁[ₖ₋₁] / Jₖ₋½
  # ξyₖ₋½ = edge_metrics.k₊½.ξ̂.x₂[ₖ₋₁] / Jₖ₋½
  # ξzₖ₋½ = edge_metrics.k₊½.ξ̂.x₃[ₖ₋₁] / Jₖ₋½

  # ηxₖ₊½ = edge_metrics.k₊½.η̂.x₁[idx] / Jₖ₊½
  # ηyₖ₊½ = edge_metrics.k₊½.η̂.x₂[idx] / Jₖ₊½
  # ηzₖ₊½ = edge_metrics.k₊½.η̂.x₃[idx] / Jₖ₊½

  # ηxₖ₋½ = edge_metrics.k₊½.η̂.x₁[ₖ₋₁] / Jₖ₋½
  # ηyₖ₋½ = edge_metrics.k₊½.η̂.x₂[ₖ₋₁] / Jₖ₋½
  # ηzₖ₋½ = edge_metrics.k₊½.η̂.x₃[ₖ₋₁] / Jₖ₋½

  # ζxₖ₊½ = edge_metrics.k₊½.ζ̂.x₁[idx] / Jₖ₊½
  # ζyₖ₊½ = edge_metrics.k₊½.ζ̂.x₂[idx] / Jₖ₊½
  # ζzₖ₊½ = edge_metrics.k₊½.ζ̂.x₃[idx] / Jₖ₊½

  # ζxₖ₋½ = edge_metrics.k₊½.ζ̂.x₁[ₖ₋₁] / Jₖ₋½
  # ζyₖ₋½ = edge_metrics.k₊½.ζ̂.x₂[ₖ₋₁] / Jₖ₋½
  # ζzₖ₋½ = edge_metrics.k₊½.ζ̂.x₃[ₖ₋₁] / Jₖ₋½

  # flux divergence

  # αᵢⱼₖ = (
  #   ξx * (ξxᵢ₊½ - ξxᵢ₋½) +
  #   ξy * (ξyᵢ₊½ - ξyᵢ₋½) +
  #   ξz * (ξzᵢ₊½ - ξzᵢ₋½) +
  #   #
  #   ηx * (ξxⱼ₊½ - ξxⱼ₋½) +
  #   ηy * (ξyⱼ₊½ - ξyⱼ₋½) +
  #   ηz * (ξzⱼ₊½ - ξzⱼ₋½) +
  #   #
  #   ζx * (ξxₖ₊½ - ξxₖ₋½) +
  #   ζy * (ξyₖ₊½ - ξyₖ₋½) +
  #   ζz * (ξzₖ₊½ - ξzₖ₋½)
  # )

  # βᵢⱼₖ = (
  #   ξx * (ηxᵢ₊½ - ηxᵢ₋½) +
  #   ξy * (ηyᵢ₊½ - ηyᵢ₋½) +
  #   ξz * (ηzᵢ₊½ - ηzᵢ₋½) +
  #   #
  #   ηx * (ηxⱼ₊½ - ηxⱼ₋½) +
  #   ηy * (ηyⱼ₊½ - ηyⱼ₋½) +
  #   ηz * (ηzⱼ₊½ - ηzⱼ₋½) +
  #   #
  #   ζx * (ηxₖ₊½ - ηxₖ₋½) +
  #   ζy * (ηyₖ₊½ - ηyₖ₋½) +
  #   ζz * (ηzₖ₊½ - ηzₖ₋½)
  # )

  # γᵢⱼₖ = (
  #   ξx * (ζxᵢ₊½ - ζxᵢ₋½) +
  #   ξy * (ζyᵢ₊½ - ζyᵢ₋½) +
  #   ξz * (ζzᵢ₊½ - ζzᵢ₋½) +
  #   #
  #   ηx * (ζxⱼ₊½ - ζxⱼ₋½) +
  #   ηy * (ζyⱼ₊½ - ζyⱼ₋½) +
  #   ηz * (ζzⱼ₊½ - ζzⱼ₋½) +
  #   #
  #   ζx * (ζxₖ₊½ - ζxₖ₋½) +
  #   ζy * (ζyₖ₊½ - ζyₖ₋½) +
  #   ζz * (ζzₖ₊½ - ζzₖ₋½)
  # )

  ∂qᵢ∂ξ = (ξx^2 + ξy^2 + ξz^2) * (qᵢ[i, j, k] - qᵢ[i - 1, j, k])
  ∂qⱼ∂η = (ηx^2 + ηy^2 + ηz^2) * (qⱼ[i, j, k] - qⱼ[i, j - 1, k])
  ∂qₖ∂ζ = (ζx^2 + ζy^2 + ζz^2) * (qₖ[i, j, k] - qₖ[i, j, k - 1])

  # # ---------------
  # # ∂/∂η (α ∂u/∂ξ) aka ∂qᵢ/∂η
  # ∂qᵢ∂η =
  #   0.5(ηx * ξx + ηy * ξy + ηz * ξz) * (
  #     #  take average and do diff in j (for ∂/∂η)
  #     0.5(qᵢ[i, j + 1, k] + qᵢ[i - 1, j + 1, k]) - #
  #     0.5(qᵢ[i, j - 1, k] + qᵢ[i - 1, j - 1, k])
  #   )

  # # ∂/∂ξ (α ∂u/∂η) aka ∂qⱼ/∂ξ
  # ∂qⱼ∂ξ =
  #   0.5(ηx * ξx + ηy * ξy + ηz * ξz) * (
  #     #  take average and do diff in i (for ∂/∂ξ)
  #     0.5(qⱼ[i, j, k] + qⱼ[i, j, k]) - #
  #     0.5(qⱼ[i, j, k] + qⱼ[i, j, k])
  #   )

  # # ---------------

  # # ∂/∂ζ (α ∂u/∂η) aka ∂qⱼ/∂ζ
  # ∂qⱼ∂ζ =
  #   0.5(ζx * ηx + ζy * ηy + ζy * ηy) * (
  #     #  take average and do diff in k (for ∂/∂ζ)
  #     0.5(qⱼ[i, j, k] + qⱼ[i, j, k]) - #
  #     0.5(qⱼ[i, j, k] + qⱼ[i, j, k])
  #   )

  # # ∂/∂η (α ∂u/∂ζ) aka ∂qₖ/∂η
  # ∂qₖ∂η =
  #   0.5(ζx * ηx + ζy * ηy + ζy * ηy) * (
  #     #  take average and do diff in j (for ∂/∂η)
  #     0.5(qₖ[i, j, k] + qₖ[i, j, k]) - #
  #     0.5(qₖ[i, j, k] + qₖ[i, j, k])
  #   )

  # # ---------------

  # # ∂/∂ζ (α ∂u/∂ξ) aka ∂qᵢ/∂ζ
  # ∂qᵢ∂ζ =
  #   0.5(ζx * ξx + ζy * ξy + ζy * ξy) * (
  #     #  take average and do diff in k (for ∂/∂ζ)
  #     0.5(qᵢ[i, j, k] + qᵢ[i, j, k]) - #
  #     0.5(qᵢ[i, j, k] + qᵢ[i, j, k])
  #   )

  # # ∂/∂ξ (α ∂u/∂ζ) aka ∂qₖ/∂ξ
  # ∂qₖ∂ξ =
  #   0.5(ζx * ξx + ζy * ξy + ζy * ξy) * (
  #     #  take average and do diff in i (for ∂/∂ξ)
  #     0.5(qₖ[i, j, k] + qₖ[i, j, k]) - #
  #     0.5(qₖ[i, j, k] + qₖ[i, j, k])
  #   )

  # ---------------

  # additional non-orthogonal terms
  # ∂q∂ξ_α = αᵢⱼₖ * 0.5(qᵢ[i, j, k] + qᵢ[i - 1, j, k])
  # ∂q∂η_β = βᵢⱼₖ * 0.5(qⱼ[i, j, k] + qⱼ[i, j - 1, k])
  # ∂q∂ζ_γ = γᵢⱼₖ * 0.5(qₖ[i, j, k] + qₖ[i, j, k - 1])

  ∇q = (
    ∂qᵢ∂ξ + ∂qⱼ∂η + ∂qₖ∂ζ # +
    # #
    # ∂qᵢ∂η +
    # ∂qᵢ∂ζ +
    # ∂qⱼ∂ξ +
    # ∂qⱼ∂ζ +
    # ∂qₖ∂η +
    # ∂qₖ∂ξ +
    # #
    # ∂q∂ξ_α +
    # ∂q∂η_β +
    # ∂q∂ζ_γ
  )
  return ∇q
end

@kernel function _update_resid!(
  resid, cell_center_metrics, edge_metrics, α, u, u_prev, q, source_term, dt, index_offset
)
  idx = @index(Global, Cartesian)
  idx += index_offset

  @inbounds begin
    @inline ∇q = flux_divergence(q, u, α, cell_center_metrics, edge_metrics, idx)

    resid[idx] = -(u[idx] - u_prev[idx]) / dt - ∇q + source_term[idx]
  end
end

"""
    update_residual!(solver::PseudoTransientSolver, mesh, Δt)

"""
function update_residual!(solver::PseudoTransientSolver, mesh, Δt)
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _update_resid!(solver.backend)(
    solver.residual,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    solver.α,
    solver.u,
    solver.u_prev,
    solver.q′,
    solver.source_term,
    Δt,
    idx_offset;
    ndrange=size(domain),
  )

  KernelAbstractions.synchronize(solver.backend)
  return nothing
end

# ------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------
function update_conductivity!(
  scheme, mesh, temperature, density, cₚ::Real, κ::F
) where {F<:Function}
  @unpack diff_domain, domain = _domain_pairs(scheme, mesh)

  α = @view scheme.α[diff_domain]
  T = @view temperature[domain]
  ρ = @view density[domain]

  backend = scheme.backend
  conductivity_kernel(backend)(α, T, ρ, cₚ, κ; ndrange=size(α))

  KernelAbstractions.synchronize(backend)

  return nothing
end

function update_conductivity!(
  scheme, mesh, temperature, density, cₚ::AbstractArray, κ::F
) where {F<:Function}
  @unpack diff_domain, domain = _domain_pairs(scheme, mesh)

  α = @view scheme.α[diff_domain]
  T = @view temperature[domain]
  _cₚ = @view cₚ[domain]
  ρ = @view density[domain]

  backend = scheme.backend
  conductivity_kernel(backend)(α, T, ρ, _cₚ, κ; ndrange=size(α))

  KernelAbstractions.synchronize(backend)

  return nothing
end

function _domain_pairs(scheme::PseudoTransientSolver, mesh)
  diff_domain = scheme.iterators.full.cartesian
  domain = mesh.iterators.cell.full

  return (; diff_domain, domain)
end

# conductivity with array-valued cₚ
@kernel function conductivity_kernel(
  α, temperature, density, cₚ::AbstractArray{T,N}, κ::F
) where {T,N,F<:Function}
  idx = @index(Global)

  @inbounds begin
    α[idx] = κ(density[idx], temperature[idx]) / (density[idx] * cₚ[idx])
  end
end

# conductivity with single-valued cₚ
@kernel function conductivity_kernel(
  α, temperature, density, cₚ::Real, κ::F
) where {F<:Function}
  idx = @index(Global)

  @inbounds begin
    α[idx] = abs(κ(density[idx], temperature[idx]) / (density[idx] * cₚ))
  end
end

@inline cutoff(a) = (0.5(abs(a) + a)) * isfinite(a)

function cutoff!(a)
  backend = KernelAbstractions.get_backend(a)
  cutoff_kernel!(backend)(a; ndrange=size(a))
  return nothing
end

@kernel function cutoff_kernel!(a)
  idx = @index(Global, Linear)

  @inbounds begin
    _a = cutoff(a[idx])
    a[idx] = _a
  end
end

end # module
