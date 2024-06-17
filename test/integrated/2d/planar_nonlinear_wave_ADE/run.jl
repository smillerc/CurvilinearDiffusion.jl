using Plots
using CurvilinearGrids, CurvilinearDiffusion
using Printf, Adapt
using TimerOutputs
using KernelAbstractions
using Glob
using LinearAlgebra

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

function wavy_grid(nx, ny)
  x0, x1 = (0, 1)
  y0, y1 = (0, 1)
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
  x0, x1 = (0, 1)
  y0, y1 = (0, 1)

  x(i, j) = @. x0 + (x1 - x0) * ((i - 1) / (nx - 1))
  y(i, j) = @. y0 + (y1 - y0) * ((j - 1) / (ny - 1))

  return (x, y)
end

function initialize_mesh()
  ni, nj = (101, 21)
  nhalo = 1
  #   x, y = wavy_grid(ni, nj)
  x, y = uniform_grid(ni, nj)
  return CurvilinearGrid2D(x, y, (ni, nj), nhalo)
end

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
function init_state()
  mesh = adapt(ArrayT, initialize_mesh())

  bcs = (
    ilo=DirichletBC(1.0),  #
    ihi=DirichletBC(1e-30),  #
    jlo=NeumannBC(),  #
    jhi=NeumannBC(),  #
  )
  solver = ADESolver(
    mesh,
    bcs;
    backend=backend,
    face_conductivity=:arithmetic, # :harmonic won't work for T=0
  )

  # Temperature and density
  T = zeros(Float64, cellsize_withhalo(mesh))
  ρ = ones(Float64, cellsize_withhalo(mesh))
  cₚ = 1.0

  # Define the conductivity model
  @inline κ(ρ, T, κ0=1) = κ0 * T^3

  return solver, mesh, adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function run(maxt, maxiter=Inf)
  casename = "planar_nonlinear_heat_wave"

  scheme, mesh, T, ρ, cₚ, κ = init_state()
  global Δt = 1e-4
  global t = 0.0
  global iter = 0
  global io_interval = 0.05
  global io_next = io_interval
  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)

  while true
    if iter == 0
      reset_timer!()
    end

    if iter >= maxiter || t >= maxt
      break
    end

    nonlinear_thermal_conduction_step!(
      scheme, mesh, T, ρ, cₚ, κ, Δt; cutoff=false, apply_density_bc=false
    )

    @printf "cycle: %i t: %.4e, Δt: %.3e\n" iter t Δt

    if t + Δt > io_next
      @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)
      global io_next += io_interval
    end

    global iter += 1
    global t += Δt
  end

  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)

  print_timer()
  return scheme, mesh, T, ρ
end

begin
  rm.(glob("*.vts"))
  cd(@__DIR__)
  @profview begin
    scheme, mesh, temperature, dens = run(1.0, 100)
  end
  #   nothing

  #   xc, yc = centroids(mesh)

  #   domain = mesh.iterators.cell.domain
  #   ddomain = scheme.iterators.domain.cartesian
  #   T = @view temperature[domain]
  #   st = @view scheme.source_term[ddomain]

  #   x = xc[:, 1]

  #   T1d = copy(T[:, 1])

  #   front_pos = [0.870571]

  #   global xfront = 0.0
  #   for i in reverse(eachindex(T1d))
  #     if T1d[i] > 1e-10
  #       global xfront = x[i]
  #       break
  #     end
  #   end

  #   f = plot(
  #     x,
  #     T1d;
  #     title="Nonlinear heat front @ t = 1",
  #     label="simulation",
  #     marker=:circle,
  #     ms=2,
  #     xticks=0:0.2:1,
  #     yticks=0:0.2:1,
  #   )
  #   vline!(front_pos; label="analytic front position", color=:black, lw=2, ls=:dash)
  #   savefig(f, "planar_nonlinear_heat_front.png")

  #   f
end