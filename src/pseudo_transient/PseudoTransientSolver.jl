module PseudoTransientScheme

using LinearAlgebra: norm

using TimerOutputs: @timeit
using CartesianDomains: expand, shift, expand_lower, haloedge_regions
using CurvilinearGrids: CurvilinearGrid2D, CurvilinearGrid3D, cellsize_withhalo, coords
using KernelAbstractions
using KernelAbstractions: CPU, GPU, @kernel, @index
using Polyester: @batch
using UnPack: @unpack
using Printf
using StaticArrays
using WriteVTK
using MPI
using NVTX

using .Threads

using ..TimeStepControl
using ..MPIDomainDecomposition
using ..BoundaryConditions

include("../averaging.jl")
include("../validity_checks.jl")
include("../edge_terms.jl")

export PseudoTransientSolver

struct PseudoTransientSolver{N,T,BE,AA<:AbstractArray{T,N},NT1,DM,B,F}
  u::AA           # solution, e.g. temperature or whatever we're diffusing
  u_prev::AA      # previous solution
  source_term::AA # external source term
  q::NT1          # flux (there are 2 versions due to the PT algorithm)
  q′::NT1         # flux (there are 2 versions due to the PT algorithm)
  res::AA         # residual
  # Re::AA          # numerial Reynolds number
  α::AA           # diffusivity
  θr_dτ::AA       # iteration parameters
  dτ_ρ::AA        # iteration parameters
  spacing::NTuple{N,T} # physical spacing, used for Re calculation
  L::T            # length scale used for Re calculation
  iterators::DM   # domain iterators, e.g. CartesianIndices
  bcs::B          # boundary conditions
  mean::F         # which mean function are we using? arithmetic or harmonic mean
  backend::BE     # CPU or GPU backend
  nhalo::Int      # halo region width (can be different than the mesh)
  # mpi_topology::MT  # parallel topology, e.g. rank, nranks, who are my neighbors?
end

function PseudoTransientSolver(
  mesh, bcs; backend=CPU(), face_diffusivity=:harmonic, T=Float64, kwargs...
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
  residual = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  # Reynolds_number = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  α = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  θr_dτ = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  dτ_ρ = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))

  # edge-based; same dims as the cell-based arrays, but each entry stores the {i,j,k}+1/2 value
  q = flux_tuple(mesh, backend, T)
  q′ = flux_tuple(mesh, backend, T)

  L, spacing = phys_dims(mesh, T)
  if face_diffusivity === :harmonic
    mean_func = harmonic_mean # from ../averaging.jl
  else
    mean_func = arithmetic_mean # from ../averaging.jl
  end

  metric_cache = nothing
  nhalo = mesh.nhalo

  return PseudoTransientSolver(
    u,
    u_prev,
    S,
    q,
    q′,
    residual,
    # Reynolds_number,
    α,
    θr_dτ,
    dτ_ρ,
    spacing,
    L,
    iterators,
    bcs,
    mean_func,
    backend,
    nhalo,
  )
end

function phys_dims(mesh::CurvilinearGrid2D, T)
  x, y = coords(mesh)
  spacing = (minimum(diff(x; dims=1)), minimum(diff(y; dims=2))) .|> T
  min_x, max_x = extrema(x)
  min_y, max_y = extrema(y)
  L = max(abs(max_x - min_x), abs(max_y - min_y)) |> T
  # L = maximum(spacing)

  return L, spacing
end

function phys_dims(mesh::CurvilinearGrid3D, T)
  x, y, z = coords(mesh)
  spacing =
    (minimum(diff(x; dims=1)), minimum(diff(y; dims=2)), minimum(diff(z; dims=3))) .|> T

  min_x, max_x = extrema(x)
  min_y, max_y = extrema(y)
  min_z, max_z = extrema(z)

  L = max(abs(max_x - min_x), abs(max_y - min_y), abs(max_z - min_z)) |> T
  # L = maximum(spacing)
  return L, spacing
end

function flux_tuple(mesh::CurvilinearGrid2D, backend, T)
  return (
    x=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    y=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
  )
end

function flux_tuple(mesh::CurvilinearGrid3D, backend, T)
  return (
    x=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    y=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    z=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
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
  dt,
  mpi_topology;
  max_iter=1e5,
  rel_tol=1e-5,
  abs_tol=1e-9,
  error_check_interval=2,
  apply_cutoff=true,
  subcycle_conductivity=true,
  kwargs...,
) where {N}

  #

  domain = solver.iterators.domain.cartesian
  nhalo = 1

  iter = 0
  rel_error = 2 * rel_tol
  abs_error = 2 * abs_tol
  init_L₂ = Inf

  CFL = 1 / sqrt(N)

  dx, dy = solver.spacing
  Vpdτ = CFL * min(dx, dy)

  @assert dt > 0
  @assert dx > 0
  @assert dy > 0
  @assert Vpdτ > 0

  copy!(solver.u, T)
  copy!(solver.u_prev, T)

  @timeit "update_conductivity!" update_conductivity!(solver, mesh, solver.u, ρ, cₚ, κ)

  @timeit "validate_scalar (α)" validate_scalar(
    solver.α, domain, nhalo, :diffusivity; enforce_positivity=true
  )

  @timeit "applybcs! (α)" applybcs!(solver.bcs, mpi_topology, mesh, solver.α)

  @timeit "updatehalo! (α)" updatehalo!(solver.α, solver.nhalo, mpi_topology)

  @timeit "validate_scalar (u)" validate_scalar(
    solver.u, domain, nhalo, :u; enforce_positivity=true
  )

  @timeit "validate_scalar (source_term)" validate_scalar(
    solver.source_term, domain, nhalo, :source_term; enforce_positivity=false
  )

  # Pseudo-transient iteration
  while true
    # println("Iteration: $iter, rank: $(mpi_topology.rank) $rel_error, $abs_error")
    iter += 1

    # Diffusion coefficient
    if subcycle_conductivity
      if iter > 1
        @timeit "update_conductivity!" update_conductivity!(
          solver, mesh, solver.u, ρ, cₚ, κ
        )
        @timeit "applybcs! (α)" applybcs!(solver.bcs, mpi_topology, mesh, solver.α)
        @timeit "updatehalo! (α)" updatehalo!(solver.α, solver.nhalo, mpi_topology)
      end
    end

    @timeit "update_iteration_params!" update_iteration_params!(solver, ρ, Vpdτ, dt)

    @timeit "applybcs! (u)" applybcs!(solver.bcs, mpi_topology, mesh, solver.u)

    @timeit "updatehalo! (u)" updatehalo!(solver.u, solver.nhalo, mpi_topology)

    @timeit "updatehalo! (θr_dτ)" updatehalo!(solver.θr_dτ, solver.nhalo, mpi_topology) # θr_dτ is needed for the flux compuation

    # @timeit "validate_scalar (θr_dτ)" validate_scalar(
    #   solver.θr_dτ, domain, nhalo, :θr_dτ; enforce_positivity=false
    # )

    # @timeit "validate_scalar (dτ_ρ)" validate_scalar(
    #   solver.dτ_ρ, domain, nhalo, :dτ_ρ; enforce_positivity=false
    # )

    @timeit "compute_flux!" compute_flux!(solver, mesh)
    @timeit "updatehalo! (flux)" begin
      updatehalo!(solver.q, solver.nhalo, mpi_topology)
      updatehalo!(solver.q′, solver.nhalo, mpi_topology)
    end
    @timeit "compute_update!" compute_update!(solver, mesh, dt)

    # Apply a cutoff function to remove negative / non-finite values
    if apply_cutoff
      @timeit "cutoff!" cutoff!(solver.u, solver.backend)
    end

    if iter % error_check_interval == 0 || iter == 1
      @timeit "update_residual!" update_residual!(solver, mesh, dt)
      # validate_scalar(solver.res, domain, nhalo, :resid; enforce_positivity=false)

      NVTX.@range "norm" begin
        @timeit "norm" begin
          inner_dom = solver.iterators.domain.cartesian
          residual = @view solver.res[inner_dom]
          L₂ = L2_norm(residual, solver.backend)

          if iter == 1
            init_L₂ = L₂
          end

          _rel_error = L₂ ./ init_L₂
          _abs_error = L₂

          rel_error = MPI.Allreduce(_rel_error, MPI.MAX, mpi_topology.comm)
          abs_error = MPI.Allreduce(_abs_error, MPI.MAX, mpi_topology.comm)
        end
      end
    end

    if !isfinite(rel_error) || !isfinite(abs_error)
      # to_vtk(solver, mesh, iter, iter)
      error(
        "Non-finite error detected! abs_error = $abs_error, rel_error = $rel_error, exiting...",
      )
    end

    if iter > max_iter
      # to_vtk(solver, mesh, iter, iter)
      error(
        "Maximum iteration limit reached ($max_iter), abs_error = $abs_error, rel_error = $rel_error, exiting...",
      )
    end

    if (rel_error <= rel_tol || abs_error <= abs_tol)
      # println("Rank $(mpi_topology.rank) is breaking out")
      break
    end

    MPI.Barrier(mpi_topology.comm)
  end

  @timeit "validate_scalar (u)" validate_scalar(
    solver.u, domain, nhalo, :u; enforce_positivity=true
  )

  @timeit "next_dt" begin
    _next_Δt = next_dt(solver.u, solver.u_prev, dt; kwargs...)
    next_Δt = MPI.Allreduce(_next_Δt, min, mpi_topology.comm)
  end

  copy!(T, solver.u)

  stats = (rel_err=rel_error, abs_err=abs_error, niter=iter)
  return stats, next_Δt
end

# ------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------

NVTX.@annotate function cutoff!(A, ::CPU)
  @batch for idx in eachindex(A)
    a = A[idx]
    A[idx] = (0.5(abs(a) + a)) * isfinite(a)
  end

  return nothing
end

NVTX.@annotate function cutoff!(A, ::GPU)
  @. A = (0.5(abs(A) + A)) * isfinite(A)
end

get_filename(iteration) = file_prefix * @sprintf("%07i", iteration)

function get_filename(iteration, name::String)
  name = replace(name, " " => "_")
  if !endswith(name, "_")
    name = name * "_"
  end
  return name * @sprintf("%07i", iteration)
end

# function to_vtk(scheme, mesh, iteration=0, t=0.0, name="diffusion", T=Float32)
#   fn = get_filename(iteration, name)
#   @info "Writing to $fn"

#   domain = mesh.iterators.cell.domain

#   _coords = Array{T}.(coords(mesh))

#   @views vtk_grid(fn, _coords...) do vtk
#     vtk["TimeValue"] = t
#     vtk["u"] = Array{T}(scheme.u[domain])
#     vtk["u_prev"] = Array{T}(scheme.u_prev[domain])
#     vtk["residual"] = Array{T}(scheme.res[domain])

#     for (i, qi) in enumerate(scheme.q)
#       vtk["q$i"] = Array{T}(qi[domain])
#     end

#     for (i, qi) in enumerate(scheme.q′)
#       vtk["q2$i"] = Array{T}(qi[domain])
#     end

#     vtk["diffusivity"] = Array{T}(scheme.α[domain])

#     vtk["dτ_ρ"] = Array{T}(scheme.dτ_ρ[domain])
#     vtk["θr_dτ"] = Array{T}(scheme.θr_dτ[domain])

#     vtk["source_term"] = Array{T}(scheme.source_term[domain])
#   end
# end

function check_result_folder() end

function to_vtk(scheme, mesh, iteration=0, t=0.0, name="diffusion", T=Float32)
  result_folder = "contour_data"

  if scheme.mpi_topology.rank == 0
    if !isdir(result_folder)
      mkdir(result_folder)
    end
    @info "Writing contour data..."
  end

  filename = get_filename(iteration, name)
  out_path = "$(result_folder)/$(filename)"

  saved_files = paraview_collection("$(result_folder)/full_sim"; append=true) do pvd
    pvtk = write_parallel_vtk(scheme, mesh, t, out_path, T)
    pvd[t] = pvtk
  end

  return saved_files
end

function write_parallel_vtk(scheme, mesh, t=0.0, filename="diffusion", T=Float32)
  rank = scheme.mpi_topology.rank
  nranks = scheme.mpi_topology.nranks
  ismain = rank == 0

  tiles = mesh.tiles
  extents = [tile for tile in tiles]

  _coords = Array{T}.(coords(mesh))
  domain = mesh.iterators.cell.domain

  vtk = pvtk_grid(
    filename,
    _coords...;
    part=rank + 1,
    nparts=nranks,
    ismain=ismain, # only rank 0 writes the header
    ghost_level=0, # don't include ghost/halo data
    extents=extents,
  )

  @views begin
    vtk["TimeValue"] = t
    vtk["u"] = Array{T}(scheme.u[domain])
    vtk["u_prev"] = Array{T}(scheme.u_prev[domain])
    vtk["residual"] = Array{T}(scheme.res[domain])

    for (i, qi) in enumerate(scheme.q)
      vtk["q$i"] = Array{T}(qi[domain])
    end

    for (i, qi) in enumerate(scheme.q′)
      vtk["q2$i"] = Array{T}(qi[domain])
    end

    vtk["diffusivity"] = Array{T}(scheme.α[domain])

    vtk["dτ_ρ"] = Array{T}(scheme.dτ_ρ[domain])
    vtk["θr_dτ"] = Array{T}(scheme.θr_dτ[domain])

    vtk["source_term"] = Array{T}(scheme.source_term[domain])
  end

  return vtk
end

end # module
