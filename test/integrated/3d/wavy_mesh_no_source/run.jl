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
function wavy_grid(ni, nj, nk)
  Lx = Ly = Lz = 4.0

  xmin = -Lx / 2
  ymin = -Ly / 2
  zmin = -Lz / 2

  Δx0 = Lx / ni
  Δy0 = Ly / nj
  Δz0 = Lz / nk

  Ax = 0.2 / Δx0
  Ay = 0.2 / Δy0
  Az = 0.2 / Δz0

  x(i, j, k) = xmin + Δx0 * ((i - 1) + Ax * sinpi((j - 1) * Δy0) * sinpi((k - 1) * Δz0))
  y(i, j, k) = ymin + Δy0 * ((j - 1) + Ay * sinpi((k - 1) * Δz0) * sinpi((i - 1) * Δx0))
  z(i, j, k) = zmin + Δz0 * ((k - 1) + Az * sinpi((i - 1) * Δx0) * sinpi((j - 1) * Δy0))

  return (x, y, z)
end

function uniform_grid(nx, ny, nz)
  x0, x1 = (-6, 6)
  y0, y1 = (-6, 6)
  z0, z1 = (-6, 6)

  x(i, j, k) = @. x0 + (x1 - x0) * ((i - 1) / (nx - 1))
  y(i, j, k) = @. y0 + (y1 - y0) * ((j - 1) / (ny - 1))
  z(i, j, k) = @. z0 + (z1 - z0) * ((k - 1) / (nz - 1))

  return (x, y, z)
end

function initialize_mesh()
  ni = nj = nk = 51
  nhalo = 4
  x, y, z = wavy_grid(ni, nj, nk)
  mesh = CurvilinearGrid3D(x, y, z, (ni, nj, nk), nhalo)

  return mesh
end

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
function init_state()
  @info "Initializing"
  mesh = adapt(ArrayT, initialize_mesh())

  bcs = (
    ilo=NeumannBC(),
    ihi=NeumannBC(),
    jlo=NeumannBC(),
    jhi=NeumannBC(),
    klo=NeumannBC(),
    khi=NeumannBC(),
  )

  solver = ImplicitScheme(mesh, bcs; backend=backend)

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
      return κ0 # * temperature^3
    end
  end

  fwhm = 0.5
  x0, y0, z0 = zeros(3)
  for idx in mesh.iterators.cell.domain
    x⃗c = centroid(mesh, idx)

    T[idx] =
      T_hot *
      exp(-(((x0 - x⃗c.x)^2) / fwhm + ((y0 - x⃗c.y)^2) / fwhm + ((z0 - x⃗c.z)^2) / fwhm)) +
      T_cold
  end

  return solver, mesh, adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function run(maxiter=Inf)
  casename = "wavy_mesh_3d_no_source"

  scheme, mesh, T, ρ, cₚ, κ = init_state()
  #   return scheme, T, mesh
  global Δt = 1e-4
  global t = 0.0
  global maxt = 0.2
  global iter = 0
  global io_interval = 0.01
  global io_next = io_interval
  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)

  @info "Running"
  while true
    if iter == 0
      reset_timer!()
    end

    @timeit "update_conductivity!" CurvilinearDiffusion.update_conductivity!(
      scheme.α, T, ρ, κ, cₚ
    )

    @timeit "solve!" L₂, ncycles, is_converged = CurvilinearDiffusion.ImplicitSchemeType.solve!(
      scheme, mesh, T, Δt
    )
    @printf "cycle: %i t: %.4e, Δt: %.3e\n" iter t Δt

    if t + Δt > io_next
      @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)
      global io_next += io_interval
    end

    if iter >= maxiter - 1
      break
    end

    if t >= maxt
      break
    end

    global iter += 1
    global t += Δt
  end

  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)

  print_timer()
  return scheme, T, mesh
end

begin
  cd(@__DIR__)
  rm.(glob("*.vts"))

  scheme, temperature, mesh = run(Inf)
  nothing
end
