using CurvilinearGrids, CurvilinearDiffusion
using WriteVTK, Printf, UnPack, Adapt
using BlockHaloArrays
using TimerOutputs
using KernelAbstractions
using Glob
using LinearAlgebra

# @static if Sys.islinux()
#   using MKL
# elseif Sys.isapple()
#   using AppleAccelerate
# end

NMAX = Sys.CPU_THREADS
BLAS.set_num_threads(NMAX)
BLAS.get_num_threads()

@show BLAS.get_config()

dev = :CPU

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
function wavy_grid(ni, nj)
  Lx = 12
  Ly = 12
  n_xy = 6
  n_yx = 6

  xmin = -Lx / 2
  ymin = -Ly / 2

  Δx0 = Lx / (ni - 1)
  Δy0 = Ly / (nj - 1)

  Ax = 0.4 / Δx0
  Ay = 0.8 / Δy0
  # Ax = 0.2 / Δx0
  # Ay = 0.4 / Δy0

  x(i, j) = xmin + Δx0 * ((i - 1) + Ax * sinpi((n_xy * (j - 1) * Δy0) / Ly))
  y(i, j) = ymin + Δy0 * ((j - 1) + Ay * sinpi((n_yx * (i - 1) * Δx0) / Lx))

  return (x, y)
end

function uniform_grid(nx, ny)
  x0, x1 = (-6, 6)
  y0, y1 = (-6, 6)

  x(i, j) = @. x0 + (x1 - x0) * ((i - 1) / (nx - 1))
  y(i, j) = @. y0 + (y1 - y0) * ((j - 1) / (ny - 1))

  return (x, y)
end

function initialize_mesh()
  ni, nj = (101, 101)
  nhalo = 6
  x, y = wavy_grid(ni, nj)
  # x, y = uniform_grid(ni, nj)
  return CurvilinearGrid2D(x, y, (ni, nj), nhalo)
end

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
function init_state()
  mesh = adapt(ArrayT, initialize_mesh())
  bcs = (
    ilo=NeumannBC(),  #
    ihi=NeumannBC(),  #
    jlo=NeumannBC(),  #
    jhi=NeumannBC(),  #
  )

  # solver = ImplicitScheme(mesh, bcs; backend=backend, direct_solve=true)
  solver = ADESolver(
    mesh,
    bcs;
    backend=backend,
    face_conductivity=:arithmetic, # :harmonic won't work for T=0
  )

  # Temperature and density
  T_hot = 1e2
  T_cold = 1e-2
  T = ones(Float64, cellsize_withhalo(mesh)) * T_cold
  ρ = ones(Float64, cellsize_withhalo(mesh))
  source_term = similar(solver.source_term)
  cₚ = 1.0

  # Define the conductivity model
  @inline function κ(ρ, temperature, κ0=10.0)
    if !isfinite(temperature)
      return 0.0
    else
      return κ0 * temperature^3
    end
  end

  # Gaussian source term
  fwhm = 0.5
  x0 = 0.0
  y0 = 0.0
  for (idx, midx) in zip(solver.iterators.domain.cartesian, mesh.iterators.cell.domain)
    x⃗c = centroid(mesh, midx)

    source_term[idx] =
      T_hot * exp(-(((x0 - x⃗c.x)^2) / fwhm + ((y0 - x⃗c.y)^2) / fwhm)) + T_cold
  end

  copy!(solver.source_term, source_term) # move to gpu (if need be)

  return solver, mesh, adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function run(maxiter=Inf)
  casename = "wavy_mesh_2d_with_source"

  scheme, mesh, T, ρ, cₚ, κ = init_state()

  global Δt = 1e-4
  global t = 0.0
  global maxt = 1.0
  global iter = 0
  global io_interval = 0.01
  global io_next = io_interval
  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)

  while true
    if iter == 0
      reset_timer!()
    end

    nonlinear_thermal_conduction_step!(scheme, mesh, T, ρ, cₚ, κ, Δt; cutoff=true)

    @printf "cycle: %i t: %.4e, Δt: %.3e\n" iter t Δt

    if t + Δt > io_next
      @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)
      global io_next += io_interval
    end

    if t >= maxt
      break
    end

    global iter += 1
    global t += Δt
    if iter >= maxiter - 1
      break
    end
  end

  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)

  print_timer()
  return scheme, mesh, T
end

begin
  cd(@__DIR__)
  rm.(glob("*.vts"))

  solver, mesh, temperature = run(Inf)
  nothing
end

# T_cpu = Array(temperature)
# using Plots: heatmap

# heatmap(T_cpu)
# solver, mesh, T, ρ, cₚ, κ = init_state();
