using CairoMakie
using CurvilinearGrids, CurvilinearDiffusion
using Printf, Adapt
using TimerOutputs
using KernelAbstractions
using Glob
using LinearAlgebra

dev = :CPU
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
    ilo=DirichletBC(1.0),  #
    ihi=DirichletBC(0.0),  #
    jlo=PeriodicBC(),  #
    jhi=PeriodicBC(),  #
  )

  # solver = ImplicitScheme(
  #   mesh,
  #   bcs;
  #   backend=backend,
  #   direct_solve=false, # either UMFPACKFactorization (direct) or Kyrlov (iterative) 
  #   face_conductivity=:arithmetic, # :harmonic won't work for T=0
  # )

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
function run(solver_scheme, maxt, maxiter=Inf; kwargs...)
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
        # subcycle_conductivity=false,
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
  scheme, mesh, temperature, dens = run(solver_scheme, 1.0, Inf;)

  xc, yc = centroids(mesh)

  domain = mesh.iterators.cell.domain
  ddomain = scheme.iterators.domain.cartesian
  T = @view temperature[domain]
  st = @view scheme.source_term[ddomain]

  x = xc[:, 1]

  T1d = copy(T[:, 1])

  front_pos = [0.870571]

  global xfront = 0.0
  for i in reverse(eachindex(T1d))
    if T1d[i] > 1e-10
      global xfront = x[i]
      break
    end
  end

  f = plot(
    x,
    T1d;
    title="Nonlinear heat front @ t = 1",
    label="simulation",
    marker=:circle,
    ms=2,
    xticks=0:0.2:1,
    yticks=0:0.2:1,
  )
  vline!(front_pos; label="analytic front position", color=:black, lw=2, ls=:dash)
  savefig(f, "planar_nonlinear_heat_front.png")

  f
end

begin
  scatter(xc, T; color=:red, label=nothing, ms=1)
  vline!(front_pos; label="analytic front position", color=:black, lw=2, ls=:dash)
end