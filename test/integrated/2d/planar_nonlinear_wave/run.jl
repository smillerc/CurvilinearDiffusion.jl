using CairoMakie
using CurvilinearGrids, CurvilinearDiffusion
using Printf, Adapt
using TimerOutputs
using KernelAbstractions
using Glob
using LinearAlgebra

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
function wavy_grid(nx, ny, nhalo)
  x2d = zeros(nx, ny)
  y2d = zeros(nx, ny)

  x1d = range(0, 1; length=nx)
  y1d = range(0, 1; length=ny)
  a0 = 0.1
  for I in CartesianIndices(x2d)
    i, j = I.I

    x = x1d[i]
    y = y1d[j]

    # x2d[i, j] = x + a0 * sinpi(2x) * cospi(2y)
    # y2d[i, j] = y + a0 * sinpi(2x) * cospi(2y)
    x2d[i, j] = x + a0 * sinpi(2x) * sinpi(2y)
    y2d[i, j] = y + a0 * sinpi(2x) * sinpi(2y)
  end

  return CurvilinearGrids.CurvilinearGrid2D(x2d, y2d, nhalo)
end

function uniform_grid(nx, ny, nhalo)
  x0, x1 = (0, 1)
  y0, y1 = (0, 1)

  return CurvilinearGrids.RectlinearGrid((x0, y0), (x1, y1), (nx, ny), nhalo, CPU(), DT)
end

function initialize_mesh()
  # ni, nj = (101, 101)
  ni, nj = (51, 51)
  nhalo = 1
  return wavy_grid(ni, nj, nhalo)
  # return uniform_grid(ni, nj, nhalo)
end

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
# Define the conductivity model
@inline κ(ρ, T, κ0=1) = κ0 * T^3

function init_state(scheme, kwargs...)
  mesh = adapt(ArrayT, initialize_mesh())

  bcs = (
    ilo=DirichletBC((; ρ=1.0, u=1.0)),  #
    ihi=DirichletBC((; ρ=0.0, u=0.0)),  #
    jlo=PeriodicBC(),  #
    jhi=PeriodicBC(),  #
    # jlo=NeumannBC(),  #
    # jhi=NeumannBC(),  #
  )

  if scheme === :implicit
    solver = ImplicitScheme(mesh, bcs; backend=backend, mean=:arithmetic, kwargs...)
  elseif scheme === :pseudo_transient
    solver = PseudoTransientSolver(
      mesh, bcs; backend=backend, mean=:arithmetic, T=DT, kwargs...
    )
  else
    error("Must choose either :implict or :pseudo_transient")
  end

  # Temperature and density
  # T = zeros(Float64, cellsize_withhalo(mesh))
  T = ones(Float64, cellsize_withhalo(mesh)) * 1e-10
  # T[1:2, :] .= 1
  ρ = ones(Float64, cellsize_withhalo(mesh))
  cₚ = 1.0

  return solver, mesh, adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function run(solver_scheme, maxt; maxiter=Inf, kwargs...)
  casename = "planar_nonlinear_heat_wave"

  scheme, mesh, T, ρ, cₚ, κ = init_state(solver_scheme, kwargs...)
  global Δt = 5e-8
  global t = 0.0
  global iter = 0
  global io_interval = 0.05
  global io_next = io_interval
  @timeit "update_conductivity!" update_conductivity!(scheme, mesh, T, ρ, cₚ, κ)
  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, ρ, mesh, iter, t, casename)

  while true
    if iter == 0
      reset_timer!()
    end

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
        apply_cutoff=false,
        show_convergence=true,
        calculate_next_dt=true,
        subcycle_conductivity=false,
        enforce_positivity=true,
        kwargs...,
      )
    end

    if t + Δt > io_next
      @timeit "save_vtk" CurvilinearDiffusion.save_vtk(
        scheme, T, ρ, mesh, iter, t, casename
      )
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
    if isfinite(next_dt)
      # Δt = min(next_dt, 1e-5)
      Δt = next_dt
    end
  end

  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, ρ, mesh, iter, t, casename)

  print_timer()
  return scheme, mesh, T, ρ
end

begin
  rm.(glob("*.vts"))
  cd(@__DIR__)

  # solver_scheme = :implicit
  solver_scheme = :pseudo_transient
  scheme, grid, temperature, dens = run(solver_scheme, 1.0; maxiter=Inf)
  nothing
end

begin
  x_analytic = [
    0.000772,
    0.0703,
    0.134,
    0.197,
    0.27,
    0.344,
    0.403,
    0.48,
    0.536,
    0.598,
    0.656,
    0.693,
    0.723,
    0.742,
    0.762,
    0.785,
    0.798,
    0.813,
    0.827,
    0.831,
    0.838,
    0.846,
    0.851,
    0.856,
    0.863,
    0.867,
    0.871,
    0.871,
  ]

  y_analytic = [
    0.999,
    0.975,
    0.952,
    0.928,
    0.896,
    0.86,
    0.83,
    0.786,
    0.746,
    0.699,
    0.648,
    0.61,
    0.575,
    0.549,
    0.52,
    0.48,
    0.455,
    0.42,
    0.385,
    0.374,
    0.352,
    0.324,
    0.297,
    0.266,
    0.211,
    0.153,
    0.0772,
    0.00368,
  ]

  xc, yc = centroids(grid) .|> Array

  domain = grid.iterators.cell.domain
  ddomain = scheme.iterators.domain.cartesian
  T = Array(temperature[domain])
  st = Array(scheme.source_term[ddomain])

  x = Array(xc[:, 1])

  T1d = copy(T[:, 1])

  front_pos = [0.870571]

  global xfront = 0.0
  for i in reverse(eachindex(T1d))
    if T1d[i] > 1e-10
      global xfront = x[i]
      break
    end
  end

  f = Figure(; size=(500, 500))
  ax = Axis(
    f[1, 1];
    aspect=1,
    xlabel="x",
    ylabel="Temperature",
    xticks=0:0.2:1,
    yticks=0:0.2:1,
    xgridvisible=false,
    ygridvisible=false,
  )

  scatter!(ax, vec(xc), vec(T); color=:red, label="Simulation", markersize=3)
  lines!(ax, x_analytic, y_analytic; color=:black, label="Analytic")
  vlines!(front_pos; label="Front Position", color=:black, linewidth=2, linestyle=:dash)
  axislegend(; position=:lb)

  save("nonlinear_heat_front.eps", f)
  display(f)
end