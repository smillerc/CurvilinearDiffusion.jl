module PseudoTransientScheme

using LinearAlgebra: norm

using TimerOutputs: @timeit
using CartesianDomains: expand, shift, expand_lower, haloedge_regions
using CurvilinearGrids: CurvilinearGrid2D, cellsize_withhalo, coords
using KernelAbstractions
using KernelAbstractions: CPU, GPU, @kernel, @index
using Polyester: @batch
using UnPack: @unpack
using Printf
using WriteVTK

using ..TimeStepControl

using ..BoundaryConditions

include("../averaging.jl")
include("../validity_checks.jl")
include("../edge_terms.jl")

export PseudoTransientSolver

struct PseudoTransientSolver{N,T,BE,AA<:AbstractArray{T,N},NT1,DM,B,F}
  H::AA
  H_prev::AA
  source_term::AA
  qH::NT1
  qH_2::NT1
  res::AA
  Re::AA
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
  mesh::CurvilinearGrid2D, bcs; backend=CPU(), face_diffusivity=:harmonic, T=Float64
)
  #
  #         H
  #     |--------|--------|--------|--------|
  #           q_i+1/2
  #

  iterators = (
    domain=(cartesian=mesh.iterators.cell.domain,),
    full=(cartesian=mesh.iterators.cell.full,),
  )

  # cell-based
  H = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  H_prev = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  S = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)) # source term

  # edge-based
  qH = (
    x=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    y=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
  )
  qH² = (
    x=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    y=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
  )

  res = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  Re = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))

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
    H,
    H_prev,
    S,
    qH,
    qH²,
    res,
    Re,
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

include("conductivity.jl")
include("flux.jl")
include("flux_divergence.jl")
include("iteration_parameters.jl")
include("residuals.jl")
include("update.jl")

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
  rel_tol=1e-5,
  abs_tol=1e-9,
  error_check_interval=10,
  cutoff=true,
  ioall_save=false,
  subcycle_conductivity=true,
  kwargs...,
) where {N}

  #
  domain = solver.iterators.domain.cartesian
  nhalo = 1

  iter = 0
  rel_err = 2 * rel_tol
  abs_err = 2 * abs_tol
  init_L₂ = Inf

  CFL = 1 / sqrt(N)

  dx, dy = solver.spacing
  Vpdτ = CFL * min(dx, dy)

  copy!(solver.H, T)
  copy!(solver.H_prev, T)

  update_conductivity!(solver, mesh, solver.H, ρ, cₚ, κ)
  # display(solver.α)
  # error("done")

  validate_scalar(solver.H, domain, nhalo, :H; enforce_positivity=true)
  validate_scalar(solver.source_term, domain, nhalo, :source_term; enforce_positivity=false)
  validate_scalar(solver.α, domain, nhalo, :diffusivity; enforce_positivity=true)

  # Pseudo-transient iteration
  # while err > tol && iter < max_iter && isfinite(err)
  while true
    # @info "Iter: $iter, Err: $err"
    iter += 1

    # Diffusion coefficient
    if subcycle_conductivity
      if iter > 1
        update_conductivity!(solver, mesh, solver.H, ρ, cₚ, κ)
      end
    end
    applybcs!(solver.bcs, mesh, solver.H)

    @timeit "update_iteration_params!" update_iteration_params!(solver, ρ, Vpdτ, dt;)

    applybcs!(solver.bcs, mesh, solver.α)
    applybcs!(solver.bcs, mesh, solver.dτ_ρ)
    applybcs!(solver.bcs, mesh, solver.θr_dτ)

    @timeit "compute_flux!" compute_flux!(solver, mesh)
    @timeit "compute_update!" compute_update!(solver, mesh, dt)

    # Apply a cutoff function to remove negative
    # if cutoff
    cutoff!(solver.H)
    # end

    @timeit "update_residual!" update_residual!(solver, mesh, dt)
    # validate_scalar(solver.res, domain, nhalo, :resid; enforce_positivity=false)

    @timeit "norm" begin
      inner_dom = solver.iterators.domain.cartesian
      residual = @view solver.res[inner_dom]
      L₂norm = L2_norm(residual)

      if iter == 1
        init_L₂ = L₂norm
      end

      rel_err = L₂norm / init_L₂
      abs_err = L₂norm
      # end
    end

    if !isfinite(rel_err) || !isfinite(abs_err)
      @show extrema(solver.residual)
      to_vtk(solver, mesh, iter, iter)
      error("Non-finite error detected! abs_err = $abs_err, rel_err = $rel_err, exiting...")
    end

    if ioall_save
      to_vtk(solver, mesh, iter, iter, "allthethings")
    end

    if iter > max_iter
      @printf "\t rel_err: %.2e, abs_err: %.2e\n" rel_err abs_err

      to_vtk(solver, mesh, iter, iter)
      error(
        "Maximum iteration limit reached ($max_iter), abs_err = $abs_err, rel_err = $rel_err, exiting...",
      )
    end

    if rel_err <= rel_tol || abs_err <= abs_tol
      # if abs_err <= abs_tol
      break
    end
  end

  # if iter == max_iter
  #   error(
  #     "Maximum iteration limit reached ($max_iter), current error is $(err), tolerance is $tol, exiting...",
  #   )
  # end

  validate_scalar(solver.H, domain, nhalo, :H; enforce_positivity=true)

  @timeit "next_dt" begin
    next_Δt = next_dt(solver.H, solver.H_prev, dt; kwargs...)
  end

  copy!(solver.H_prev, solver.H)
  copy!(T, solver.H)

  stats = (rel_err=min(rel_err, abs_err), niter=iter)
  return stats, next_Δt
end

# ------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------

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

get_filename(iteration) = file_prefix * @sprintf("%07i", iteration)

function get_filename(iteration, name::String)
  name = replace(name, " " => "_")
  if !endswith(name, "_")
    name = name * "_"
  end
  return name * @sprintf("%07i", iteration)
end

function to_vtk(scheme, mesh, iteration=0, t=0.0, name="diffusion", T=Float32)
  fn = get_filename(iteration, name)
  @info "Writing to $fn"

  domain = mesh.iterators.cell.domain

  _coords = Array{T}.(coords(mesh))

  @views vtk_grid(fn, _coords...) do vtk
    vtk["TimeValue"] = t
    vtk["H"] = Array{T}(scheme.H[domain])
    vtk["H_prev"] = Array{T}(scheme.H_prev[domain])
    vtk["residual"] = Array{T}(scheme.res[domain])
    vtk["qi"] = Array{T}(scheme.qH[1][domain])
    vtk["qj"] = Array{T}(scheme.qH[2][domain])
    vtk["q2i"] = Array{T}(scheme.qH_2[1][domain])
    vtk["q2j"] = Array{T}(scheme.qH_2[2][domain])
    vtk["diffusivity"] = Array{T}(scheme.α[domain])

    vtk["dτ_ρ"] = Array{T}(scheme.dτ_ρ[domain])
    vtk["θr_dτ"] = Array{T}(scheme.θr_dτ[domain])

    vtk["source_term"] = Array{T}(scheme.source_term[domain])
  end
end

end # module
