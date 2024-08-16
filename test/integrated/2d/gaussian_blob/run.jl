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
function wavy_grid(ni, nj, nhalo)
  Lx = 12
  Ly = 12
  n_xy = 6
  n_yx = 6

  xmin = -Lx / 2
  ymin = -Ly / 2

  Δx0 = Lx / (ni - 1)
  Δy0 = Ly / (nj - 1)

  # Ax = 0.4 / Δx0
  # Ay = 0.8 / Δy0
  Ax = 0.2 / Δx0
  Ay = 0.4 / Δy0

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

function initialize_mesh(DT)
  ni, nj = (2000, 2000)
  nhalo = 1
  # return wavy_grid(ni, nj, nhalo)
  return uniform_grid(ni, nj, nhalo)
end

function init_state_no_source(scheme, kwargs...)
  mesh = initialize_mesh(DT)

  bcs = (ilo=NeumannBC(), ihi=NeumannBC(), jlo=NeumannBC(), jhi=NeumannBC())

  if scheme === :implicit
    solver = ImplicitScheme(mesh, bcs; backend=backend, kwargs...)
  elseif scheme === :pseudo_transient
    solver = PseudoTransientSolver(mesh, bcs; backend=backend, kwargs...)
  else
    error("Must choose either :implict or :pseudo_transient")
  end
  # solver = ADESolver(mesh, bcs; backend=backend, face_conductivity=:harmonic)

  # Temperature and density
  T_cold = 1e-2
  T = ones(DT, cellsize_withhalo(mesh)) * T_cold
  ρ = ones(DT, cellsize_withhalo(mesh))
  cₚ = 1.0

  # Define the conductivity model
  @inline function κ(ρ, temperature, κ0=1.0)
    if !isfinite(temperature)
      return 0.0
    else
      return κ0 # * temperature^3
    end
  end

  fwhm = 1.0
  x0 = 0.0
  y0 = 0.0
  xc = Array(mesh.centroid_coordinates.x)
  yc = Array(mesh.centroid_coordinates.y)
  for idx in mesh.iterators.cell.domain
    T[idx] = exp(-(((x0 - xc[idx])^2) / fwhm + ((y0 - yc[idx])^2) / fwhm)) #+ T_cold
  end

  copy!(solver.u, T)
  return solver,
  adapt(ArrayT, initialize_mesh(DT)), adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ,
  κ
end

function init_state_with_source(scheme, kwargs...)
  mesh = initialize_mesh(DT)
  bcs = (ilo=NeumannBC(), ihi=NeumannBC(), jlo=NeumannBC(), jhi=NeumannBC())

  if scheme === :implicit
    solver = ImplicitScheme(mesh, bcs; backend=backend, kwargs...)
  elseif scheme === :pseudo_transient
    solver = PseudoTransientSolver(mesh, bcs; backend=backend, T=DT, kwargs...)
  else
    error("Must choose either :implict or :pseudo_transient")
  end

  # Temperature and density
  @show DT
  T_hot = 1e2 |> DT
  T_cold = 1e-2 |> DT
  T = ones(DT, cellsize_withhalo(mesh)) * T_cold
  ρ = ones(DT, cellsize_withhalo(mesh))
  source_term = zeros(DT, cellsize_withhalo(mesh))
  cₚ = 1.0 |> DT

  # Define the conductivity model
  @inline function κ(ρ, temperature)
    if !isfinite(temperature)
      return zero(ρ)
    else
      return temperature^3
    end
  end

  fwhm = 1.0 |> DT
  x0 = 0.0 |> DT
  y0 = 0.0 |> DT
  xc = Array(mesh.centroid_coordinates.x)
  yc = Array(mesh.centroid_coordinates.y)
  for idx in mesh.iterators.cell.domain
    source_term[idx] =
      T_hot * exp(-(((x0 - xc[idx])^2) / fwhm + ((y0 - yc[idx])^2) / fwhm)) + T_cold
  end

  copy!(solver.source_term, source_term)
  return solver,
  adapt(ArrayT, initialize_mesh(DT)), adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ,
  κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function solve_prob(scheme, case=:no_source, maxiter=Inf; kwargs...)
  casename = "blob"

  if case === :no_source
    scheme, mesh, T, ρ, cₚ, κ = init_state_no_source(scheme, kwargs...)
  else
    scheme, mesh, T, ρ, cₚ, κ = init_state_with_source(scheme, kwargs...)
  end

  global Δt = 1e-4
  global t = 0.0
  global maxt = 0.6
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
    @timeit "nonlinear_thermal_conduction_step!" begin
      stats, next_dt = nonlinear_thermal_conduction_step!(
        scheme, mesh, T, ρ, cₚ, κ, DT(Δt); cutoff=true, show_convergence=true
      )
    end

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

  scheme, mesh, temperature = solve_prob(:pseudo_transient, :no_source, 100)
  # scheme, mesh, temperature = solve_prob(:implicit, :no_source, 100; direct_solve=false)
  # scheme, mesh, temperature = solve_prob(:implicit, :no_source, 10; direct_solve=true)

  # @profview begin
  # scheme, mesh, temperature = solve_prob(
  #   :pseudo_transient, :with_source, 100; error_check_interval=2
  # )
  # end
  # scheme, mesh, temperature = solve_prob(:implicit, :with_source, 100; direct_solve=false)
  nothing
end
