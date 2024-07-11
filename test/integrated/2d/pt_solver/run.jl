using CurvilinearGrids, CurvilinearDiffusion
using Printf, Adapt
using TimerOutputs
using KernelAbstractions
using Glob
using LinearAlgebra
using NVTX

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

function uniform_grid(nx, ny, nhalo)
  x0, x1 = (0, 1)
  y0, y1 = (0, 0.25)
  # x0, x1 = (-5, 5)
  # y0, y1 = (-5, 5)
  # x0, x1 = (0, 10)
  # y0, y1 = (0, 10)

  return CurvilinearGrids.RectlinearGrid((x0, y0), (x1, y1), (nx, ny), nhalo)
end

function initialize_mesh()
  ni, nj = (150, 150)
  nhalo = 1
  return wavy_grid(ni, nj, nhalo)
  # return uniform_grid(ni, nj, nhalo)
end

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
function init_state()
  mesh = adapt(ArrayT, initialize_mesh())
  # mesh = initialize_mesh()

  bcs = (
    ilo=NeumannBC(),  #
    ihi=NeumannBC(),  #
    # ilo=DirichletBC(1.0),  #
    # ihi=DirichletBC(0.0),  #
    jlo=NeumannBC(),  #
    jhi=NeumannBC(),  #
  )

  solver = PseudoTransientSolver(mesh, bcs; backend=backend)

  # Temperature and density
  T_hot = 1e3
  T_cold = 1e-2
  T = ones(size(mesh.iterators.cell.full)) * T_cold
  ρ = ones(size(mesh.iterators.cell.full))
  source_term = ones(size(mesh.iterators.cell.full))
  cₚ = 1.0

  # Define the conductivity model
  @inline function κ(ρ, temperature, κ0=1.0)
    kappa = κ0 * temperature #^3

    if !isfinite(kappa)
      return zero(temperature)
    else
      return kappa
    end
  end

  fwhm = 1.0
  x0 = 0.0
  y0 = 0.0
  xc = Array(mesh.centroid_coordinates.x)
  yc = Array(mesh.centroid_coordinates.y)
  for idx in mesh.iterators.cell.domain
    T[idx] = exp(-(((x0 - xc[idx])^2) / fwhm + ((y0 - yc[idx])^2) / fwhm)) #+ T_cold

    # source_term[idx] = T_hot * exp(-(((x0 - xc[idx])^2) / fwhm + ((y0 - yc[idx])^2) / fwhm)) #+ T_cold
  end

  # copy!(solver.source_term, source_term) # move to gpu (if need be)

  return solver, mesh, adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function run(maxiter=Inf)
  casename = "pt_uniform_2d_no_source"

  scheme, mesh, T, ρ, cₚ, κ = init_state()

  global Δt = 1e-6
  #   global Δt = 0.02
  global t = 0.0
  global maxt = 0.6

  global iter = 0
  global io_interval = 0.01
  global io_next = io_interval

  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)

  while true
    if iter == 0
      reset_timer!()
    end

    @printf "cycle: %i t: %.4e, Δt: %.3e\n" iter t Δt
    err, subiter = CurvilinearDiffusion.PseudoTransientScheme.step!(
      scheme, mesh, T, ρ, cₚ, κ, Δt; max_iter=50, tol=1e-8, error_check_interval=2
    )
    @printf "\tL2: %.3e, %i\n" err subiter

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

  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(
    scheme, scheme.u, mesh, iter, t, casename
  )

  print_timer()
  return scheme, mesh, T
end

begin
  cd(@__DIR__)
  rm.(glob("*.vts"))

  # NVTX.@range "my message" begin
  scheme, mesh, temperature = run(Inf)
  nothing
  # end
end
