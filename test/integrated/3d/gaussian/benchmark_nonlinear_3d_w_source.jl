using CurvilinearGrids, CurvilinearDiffusion
using Printf, Adapt
using TimerOutputs
using KernelAbstractions
using Glob
using LinearAlgebra
using JSON3

# @static if Sys.islinux()
#   using MKL
# elseif Sys.isapple()
#   using AppleAccelerate
# end

# NMAX = Sys.CPU_THREADS
# BLAS.set_num_threads(NMAX)
# BLAS.get_num_threads()

# @show BLAS.get_config()

dev = :GPU
const DT = Float64

if dev === :GPU
  @info "Using CUDA"
  using CUDA
  using CUDA.CUDAKernels
  backend = CUDABackend()
  ArrayT = CuArray
  # CUDA.allowscalar(false)
else
  backend = CPU()
  ArrayT = Array
end

# ------------------------------------------------------------
# Grid Construction
# ------------------------------------------------------------
function wavy_grid(ni, nj, nk, nhalo)
  Lx = Ly = Lz = 12

  xmin = -Lx / 2
  ymin = -Ly / 2
  zmin = -Lz / 2

  Δx0 = Lx / ni
  Δy0 = Ly / nj
  Δz0 = Lz / nk

  Ax = 0.2 / Δx0
  Ay = 0.2 / Δy0
  Az = 0.2 / Δz0

  x = zeros(ni, nj, nk)
  y = zeros(ni, nj, nk)
  z = zeros(ni, nj, nk)

  n = 0.5
  for k in 1:nk
    for j in 1:nj
      for i in 1:ni
        x[i, j, k] =
          xmin + Δx0 * ((i - 1) + Ax * sinpi(n * (j - 1) * Δy0) * sinpi(n * (k - 1) * Δz0))
        y[i, j, k] =
          ymin + Δy0 * ((j - 1) + Ay * sinpi(n * (k - 1) * Δz0) * sinpi(n * (i - 1) * Δx0))
        z[i, j, k] =
          zmin + Δz0 * ((k - 1) + Az * sinpi(n * (i - 1) * Δx0) * sinpi(n * (j - 1) * Δy0))
      end
    end
  end

  return CurvilinearGrid3D(x, y, z, nhalo)
end

function uniform_grid(nx, ny, nz, nhalo)
  x0, x1 = (-6, 6)
  y0, y1 = (-6, 6)
  z0, z1 = (-6, 6)

  return CurvilinearGrids.RectlinearGrid((x0, y0, z0), (x1, y1, z1), (nx, ny, nz), nhalo)
end

function initialize_mesh(n)
  nhalo = 4
  @show n
  return wavy_grid(n, n, n, nhalo)
  # return uniform_grid(n, n, n, nhalo)
end

function init_state_with_source(scheme, resolution, kwargs...)

  # Define the conductivity model
  @inline function κ(ρ, temperature)
    if !isfinite(temperature)
      return zero(ρ)
    else
      return 2.5 * abs(temperature)^3
    end
  end

  mesh = initialize_mesh(resolution)
  bcs = (
    ilo=NeumannBC(), #
    ihi=NeumannBC(), #
    jlo=NeumannBC(), #
    jhi=NeumannBC(), #
    klo=NeumannBC(), #
    khi=NeumannBC(), #
  )
  if scheme === :implicit
    solver = ImplicitScheme(mesh, bcs; backend=backend, kwargs...)
  elseif scheme === :pseudo_transient
    solver = PseudoTransientSolver(mesh, bcs; backend=backend, T=DT, kwargs...)
  else
    error("Must choose either :implict or :pseudo_transient")
  end

  # Temperature and density
  T_hot = 1e4 |> DT
  T_cold = 1e-2 |> DT
  T = ones(DT, cellsize_withhalo(mesh)) * T_cold
  ρ = ones(DT, cellsize_withhalo(mesh))
  source_term = zeros(DT, cellsize_withhalo(mesh))
  cₚ = 1.0 |> DT

  fwhm = 1.0 |> DT
  x0 = y0 = z0 = 0.0 |> DT

  for idx in mesh.iterators.cell.domain
    x⃗c = centroid(mesh, idx)

    source_term[idx] =
      T_hot *
      exp(-(((x0 - x⃗c.x)^2) / fwhm + ((y0 - x⃗c.y)^2) / fwhm + ((z0 - x⃗c.z)^2) / fwhm)) +
      T_cold
  end

  if scheme === :implicit
    s1 = @view solver.source_term[solver.iterators.domain.cartesian]
    s2 = source_term[mesh.iterators.cell.domain] # make a copy since copy! doesn't work with cpu views to gpu views (by design)
    copy!(s1, s2)

  elseif scheme === :pseudo_transient
    copy!(solver.source_term, source_term)
  end

  return (
    solver,
    adapt(ArrayT, initialize_mesh(resolution)),
    adapt(ArrayT, T),
    adapt(ArrayT, ρ),
    cₚ,
    κ,
  )
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function solve_prob(scheme, case, resolution; maxiter=Inf, maxt=0.2, kwargs...)
  casename = "blob"

  if case === :no_source
    scheme, mesh, T, ρ, cₚ, κ = init_state_no_source(scheme, resolution, kwargs...)
  else
    scheme, mesh, T, ρ, cₚ, κ = init_state_with_source(scheme, resolution, kwargs...)
  end

  # global Δt = 1e-8
  global Δt = 5e-6
  global t = 0.0
  global iter = 0
  global io_interval = 0.01
  global io_next = io_interval
  @timeit "update_conductivity!" update_conductivity!(scheme, mesh, T, ρ, cₚ, κ)
  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, ρ, mesh, iter, t, casename)

  while true
    @printf "cycle: %i t: %.4e, Δt: %.3e\n" iter t Δt
    @timeit "nonlinear_thermal_conduction_step!" begin
      stats, next_dt = nonlinear_thermal_conduction_step!(
        scheme,
        mesh,
        T,
        ρ,
        cₚ,
        κ,
        DT(Δt);
        apply_cutoff=true,
        show_convergence=true,
        precon_iter_threshold=1,
        kwargs...,
      )
    end

    # if t + Δt > io_next
    #   @timeit "save_vtk" CurvilinearDiffusion.save_vtk(
    #     scheme, T, ρ, mesh, iter, t, casename
    #   )
    #   global io_next += io_interval
    # end

    if iter == 0
      reset_timer!()
    end

    if t >= maxt
      break
    end

    global iter += 1
    global t += Δt
    if iter >= maxiter - 1
      break
    end
    # Δt = min(next_dt, 1e-4)
  end

  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, ρ, mesh, iter, t, casename)

  print_timer()
  return scheme, mesh, T
end

# @profview 
function benchmark()
  cd(@__DIR__)
  rm.(glob("*.vts"))

  if !isdir("benchmark_results")
    mkdir("benchmark_results")
  end

  for scheme_name in (
    :pseudo_transient,
    # :implicit
  )
    for resolution in (
      51,
      # 101,
      # 201
    )
      reset_timer!()
      scheme, mesh, temperature = solve_prob(
        scheme_name,
        :with_source,
        resolution;
        maxiter=Inf,
        # maxt=1.5e-3, use this
        maxt=4e-3,
        direct_solve=false,
        mean=:arithmetic,
        apply_cutoff=true,
        enforce_positivity=true,
        # error_check_interval=2,
        CFL=0.4,
        subcycle_conductivity=false,
      )

      open(
        "$(@__DIR__)/benchmark_results/nonlinear_$(dev)_timing_$(resolution)_$(scheme_name).json",
        "w",
      ) do io
        JSON3.pretty(io, TimerOutputs.todict(TimerOutputs.DEFAULT_TIMER))
      end

      GC.gc() # free gpu memory
    end
  end
  nothing
end

# @profview 
benchmark()