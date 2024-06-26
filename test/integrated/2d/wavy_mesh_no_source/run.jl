using CurvilinearGrids, CurvilinearDiffusion
using Printf, Adapt
using TimerOutputs
using KernelAbstractions
using Glob
using LinearAlgebra

@static if Sys.islinux()
  using MKL
elseif Sys.isapple()
  using AppleAccelerate
end

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
function wavy_grid(ni, nj, nhalo)
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

  x = zeros(ni, nj)
  y = zeros(ni, nj)
  for j in 1:nj
    for i in 1:ni
      x[i, j] = xmin + Δx0 * ((i - 1) + Ax * sinpi((n_xy * (j - 1) * Δy0) / Ly))
      y[i, j] = ymin + Δy0 * ((j - 1) + Ay * sinpi((n_yx * (i - 1) * Δx0) / Lx))
    end
  end

  return CurvilinearGrid2D(x, y, nhalo)
end

# function wavy_grid(ni, nj, nhalo)
#   function init(ni, nj)
#     Lx = 12
#     Ly = 12
#     n_xy = 6
#     n_yx = 6

#     xmin = -Lx / 2
#     ymin = -Ly / 2

#     Δx0 = Lx / (ni - 1)
#     Δy0 = Ly / (nj - 1)

#     Ax = 0.4 / Δx0
#     Ay = 0.8 / Δy0
#     # Ax = 0.2 / Δx0
#     # Ay = 0.4 / Δy0

#     x(i, j) = xmin + Δx0 * ((i - 1) + Ax * sinpi((n_xy * (j - 1) * Δy0) / Ly))
#     y(i, j) = ymin + Δy0 * ((j - 1) + Ay * sinpi((n_yx * (i - 1) * Δx0) / Lx))

#     return (x, y)
#   end

#   x, y = init(ni, nj)
#   return CurvilinearGrid2D(x, y, (ni, nj), nhalo)
# end

function uniform_grid(nx, ny, nhalo)
  x0, x1 = (-6, 6)
  y0, y1 = (-6, 6)

  return CurvilinearGrids.RectlinearGrid((x0, y0), (x1, y1), (nx, ny), nhalo)
end

function initialize_mesh()
  ni, nj = (150, 150)
  nhalo = 6
  # return wavy_grid(ni, nj, nhalo)
  return uniform_grid(ni, nj, nhalo)
end

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
function init_state()
  # mesh = adapt(ArrayT, initialize_mesh())
  mesh = initialize_mesh()

  bcs = (
    ilo=NeumannBC(),  #
    ihi=NeumannBC(),  #
    jlo=NeumannBC(),  #
    jhi=NeumannBC(),  #
    # jhi=DirichletBC(100.0),  #
  )
  solver = ImplicitScheme(
    mesh,
    bcs;
    backend=backend,
    face_conductivity=:harmonic, #
    direct_solve=false, #
  )
  # solver = ADESolver(mesh, bcs; backend=backend, face_conductivity=:harmonic)
  # solver = ADESolverNSweep(mesh, bcs; backend=backend, face_conductivity=:harmonic)

  # Temperature and density
  T_hot = 1e3
  T_cold = 1e-2
  T = ones(Float64, cellsize_withhalo(mesh)) * T_cold
  ρ = ones(Float64, cellsize_withhalo(mesh))
  cₚ = 1.0

  # Define the conductivity model
  @inline function κ(ρ, temperature, κ0=10.0)
    if !isfinite(temperature)
      return 0.0
    else
      return κ0 * temperature^3
    end
  end

  fwhm = 0.5
  x0 = 0.0
  y0 = 0.0
  for idx in mesh.iterators.cell.domain
    x⃗c = centroid(mesh, idx)

    T[idx] = T_hot * exp(-(((x0 - x⃗c.x)^2) / fwhm + ((y0 - x⃗c.y)^2) / fwhm)) + T_cold
  end

  return solver, mesh, adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function run(maxiter=Inf)
  casename = "wavy_mesh_2d_no_source"

  scheme, mesh, T, ρ, cₚ, κ = init_state()

  # global Δt = 1e-4
  global Δt = 5e-5
  global t = 0.0
  global maxt = 0.01
  # global maxt = 0.005
  global iter = 0
  global io_interval = 0.01
  global io_next = io_interval
  @timeit "update_conductivity!" update_conductivity!(scheme, mesh, T, ρ, cₚ, κ)
  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)

  while true
    if iter == 0
      reset_timer!()
    end

    @printf "cycle: %i t: %.4e, Δt: %.3e\n" iter t Δt
    next_dt = nonlinear_thermal_conduction_step!(scheme, mesh, T, ρ, cₚ, κ, Δt; cutoff=true)

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
    # Δt = next_dt
  end

  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)

  print_timer()
  return scheme, mesh, T
end

begin
  cd(@__DIR__)
  rm.(glob("*.vts"))

  scheme, mesh, temperature = run(10)
  nothing
end
