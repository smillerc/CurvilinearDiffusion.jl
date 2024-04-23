using CurvilinearGrids, CurvilinearDiffusion
using UnPack, Adapt
using BlockHaloArrays
using TimerOutputs
using KernelAbstractions
using Glob, Printf
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

dev = :GPU

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
function wavy_grid(nx, ny)
  x0, x1 = (-6, 6)
  y0, y1 = (-6, 6)
  a0 = 0.1

  function x(i, j)
    x1d = x0 + (x1 - x0) * ((i - 1) / (nx - 1))
    y1d = y0 + (y1 - y0) * ((j - 1) / (ny - 1))
    return x1d + a0 * sin(2 * pi * x1d) * sin(2 * pi * y1d)
  end

  function y(i, j)
    x1d = x0 + (x1 - x0) * ((i - 1) / (nx - 1))
    y1d = y0 + (y1 - y0) * ((j - 1) / (ny - 1))
    return y1d + a0 * sin(2 * pi * x1d) * sin(2 * pi * y1d)
  end

  return (x, y)
end

function wavy_grid2(ni, nj)
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
  ni, nj = (501, 501)
  nhalo = 6
  x, y = wavy_grid2(ni, nj)
  # x, y = uniform_grid(ni, nj)
  return CurvilinearGrid2D(x, y, (ni, nj), nhalo)
end

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
function init_state(; direct_solve=true)
  mesh = adapt(ArrayT, initialize_mesh())

  T0 = 1.0
  bcs = (ilo=:zero_flux, ihi=:zero_flux, jlo=:zero_flux, jhi=:zero_flux)

  solver = ImplicitScheme(mesh, bcs; backend=backend, direct_solve=direct_solve)
  CFL = 100 # 1/2 is the explicit stability limit

  # Temperature and density
  T_hot = 1e3
  T_cold = 1e-2
  T = ones(Float64, cellsize_withhalo(mesh)) * T_cold
  ρ = ones(Float64, cellsize_withhalo(mesh))
  source_term = similar(ρ)
  fill!(source_term, 0)
  cₚ = 1.0

  # Define the conductivity model
  @inline function κ(ρ, temperature, κ0=1.0)
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
  for idx in mesh.iterators.cell.domain
    x⃗c = centroid(mesh, idx)

    source_term[idx] =
      T_hot * exp(-(((x0 - x⃗c.x)^2) / fwhm + ((y0 - x⃗c.y)^2) / fwhm)) + T_cold
  end

  copy!(solver.source_term, source_term) # move to gpu (if need be)

  return solver, mesh, adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function run(maxiter, params)
  solver, mesh, T, ρ, cₚ, κ = params

  casename = "diffusion123"
  global Δt0 = 1e-4
  global Δt = 1e-5
  global t = 0.0
  global maxt = 0.2
  global iter = 0
  # global maxiter = 2
  global io_interval = 5e-4
  global io_next = io_interval
  @timeit "save_vtk" save_vtk(solver, T, mesh, iter, t, casename)

  @timeit "update_conductivity!" CurvilinearDiffusion.update_conductivity!(
    solver.α, T, ρ, κ, cₚ
  )

  @profview while true
    if iter == 1
      reset_timer!()
    end

    # dt = CurvilinearDiffusion.max_dt(solver, mesh)
    # dt = max(1e-10, min(Δt0, dt))
    # global Δt = CFL * dt
    # @show Δt, CFL, dt

    @timeit "solve!" CurvilinearDiffusion.ImplicitSchemeType.solve!(solver, mesh, T, Δt)
    # @timeit "solve!" L₂, ncycles, is_converged = CurvilinearDiffusion.ImplicitSchemeType.solve!(
    #   solver, mesh, T, Δt
    # )
    # @printf "cycle: %i t: %.4e, L2: %.1e, iterations: %i Δt: %.3e\n" iter t L₂ ncycles Δt
    @printf "cycle: %i t: %.4e Δt: %.3e\n" iter t Δt

    if t + Δt > io_next
      @timeit "save_vtk" save_vtk(solver, T, mesh, iter, t, casename)
      global io_next += io_interval
    end

    global iter += 1
    global t += Δt

    if iter >= maxiter
      break
    end

    if t >= maxt
      break
    end
  end

  save_vtk(solver, T, mesh, iter, t, casename)

  print_timer()
  return solver, T, mesh
end

begin
  cd(@__DIR__)
  rm.(glob("*.vts"))
  solver, mesh, T, ρ, cₚ, κ = init_state(; direct_solve=false)

  run(50, (solver, mesh, T, ρ, cₚ, κ))
  nothing
end
