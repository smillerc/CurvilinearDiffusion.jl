using CurvilinearGrids, CurvilinearDiffusion
using WriteVTK, Printf, UnPack, Adapt
using BlockHaloArrays
using TimerOutputs
using KernelAbstractions
using Glob
using LinearAlgebra

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
  x0, x1 = (-6, 6)
  y0, y1 = (-6, 6)

  return CurvilinearGrids.RectlinearGrid((x0, y0), (x1, y1), (nx, ny), nhalo)
end

function initialize_mesh(uniform)
  ni, nj = (150, 150)
  nhalo = 6
  if uniform
    return uniform_grid(ni, nj, nhalo)
  else
    return wavy_grid(ni, nj, nhalo)
  end
end

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
function init_state(dev, uniform, direct_solve)
  mesh = initialize_mesh(uniform)
  bcs = (
    ilo=NeumannBC(),  #
    ihi=NeumannBC(),  #
    jlo=NeumannBC(),  #
    jhi=NeumannBC(),  #
  )

  solver = ImplicitScheme(mesh, bcs; backend=backend, direct_solve=direct_solve)

  # Temperature and density
  T_hot = 1e2
  T_cold = 1e-2
  T = ones(Float64, cellsize_withhalo(mesh)) * T_cold
  ρ = ones(Float64, cellsize_withhalo(mesh))
  source_term = zeros(size(solver.source_term))
  cₚ = 1.0

  # Define the conductivity model
  @inline function κ(ρ, temperature, κ0=1.0)
    if !isfinite(temperature)
      return 0.0
    else
      return κ0 #* temperature^3
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

  return solver, adapt(ArrayT, mesh), adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function solve_prob(device, uniform, direct_solve, maxiter)
  casename = "wavy_mesh_2d_with_source"

  scheme, mesh, T, ρ, cₚ, κ = init_state(device, uniform, direct_solve)

  global Δt = 1e-4
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
    stats, next_Δt = nonlinear_thermal_conduction_step!(scheme, mesh, T, ρ, cₚ, κ, Δt)

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

    global Δt = next_Δt
  end

  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)

  print_timer()
  return scheme, mesh, T
end

function run(device::Symbol, uniform::Bool, direct_solve::Bool)
  cd(@__DIR__)
  rm.(glob("*.vts"))

  solver, mesh, temperature = solve_prob(device, uniform, direct_solve, 50)
end

dev = :CPU

if dev === :GPU
  @info "Using CUDA"
  using CUDA
  using CUDA.CUDAKernels
  backend = CUDABackend()
  ArrayT = CuArray
  CUDA.allowscalar(false)
else
  @info "Using CPU"
  backend = CPU()
  ArrayT = Array
end

run(dev, false, true);