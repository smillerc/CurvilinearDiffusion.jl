using Plots
using CurvilinearGrids, CurvilinearDiffusion
using Printf, Adapt
using TimerOutputs
using KernelAbstractions
using Glob
using LinearAlgebra

const π² = π * π
const q0 = 2
u_initial(x, t) = sinpi(3x)
# u_initial(x, t) = 0.25sinpi(2x) + sinpi(3x)
q(x, t) = q0 * sinpi(x)

function u_analytic(x, t)
  sinpi(3x) * exp(-9π² * t) + (q0 / π²) * sinpi(x) * (1 - exp(-π² * t))
end
# function u_analytic(x, t)
#   0.25sinpi(2x) * exp(-4π² * t) +
#   sinpi(3x) * exp(-9π² * t) +
#   (q0 / π²) * sinpi(x) * (1 - exp(-π² * t))
# end

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
function uniform_grid(nx, ny)
  x0, x1 = (0, 1)
  y0, y1 = (0, 1)

  x(i, j) = @. x0 + (x1 - x0) * ((i - 1) / (nx - 1))
  y(i, j) = @. y0 + (y1 - y0) * ((j - 1) / (ny - 1))

  return (x, y)
end

function initialize_mesh()
  ni, nj = (101, 101)
  nhalo = 4
  # x, y = wavy_grid(ni, nj)
  x, y = uniform_grid(ni, nj)
  return CurvilinearGrid2D(x, y, (ni, nj), nhalo)
end

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
function init_state()
  mesh = adapt(ArrayT, initialize_mesh())

  bcs = (
    ilo=DirichletBC(0.0),  #
    ihi=DirichletBC(0.0),  #
    jlo=NeumannBC(),  #
    jhi=NeumannBC(),  #
  )
  solver = ImplicitScheme(mesh, bcs; backend=backend)

  # Temperature and density
  T = zeros(Float64, cellsize_withhalo(mesh))
  source_term = zeros(Float64, cellsize_withhalo(mesh))
  ρ = ones(Float64, cellsize_withhalo(mesh))
  cₚ = 1.0

  # Define the conductivity model
  @inline κ(ρ, temperature) = 1.0

  for idx in mesh.iterators.cell.domain
    x⃗c = centroid(mesh, idx)
    T[idx] = u_initial(x⃗c.x, 0.0)
    source_term[idx] = q(x⃗c.x, 0.0)
  end

  scheme_q = @view solver.source_term[solver.iterators.domain.cartesian]
  mesh_q = @view source_term[mesh.iterators.cell.domain]
  copy!(scheme_q, mesh_q)

  #   fill!(solver.source_term, 0.0)
  fill!(solver.α, 1.0)

  return solver, mesh, adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function run(maxt, maxiter=Inf)
  casename = "multimode_sine"

  scheme, mesh, T, ρ, cₚ, κ = init_state()
  global Δt = 5e-5
  global t = 0.0
  global iter = 0
  global io_interval = 0.01
  global io_next = io_interval
  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)

  while true
    if iter == 0
      reset_timer!()
    end

    if iter >= maxiter || t >= maxt
      break
    end

    nonlinear_thermal_conduction_step!(scheme, mesh, T, ρ, cₚ, κ, Δt; cutoff=false)

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
  return scheme, mesh, T
end

begin
  cd(@__DIR__)
  rm.(glob("*.vts"))
  tfinal = 0.05
  scheme, mesh, temperature = run(tfinal, Inf)
  nothing

  xc, yc = centroids(mesh)

  domain = mesh.iterators.cell.domain
  ddomain = scheme.iterators.domain.cartesian
  T = @view temperature[domain]
  st = @view scheme.source_term[ddomain]

  #   x = 0:0.01:1
  x = xc[:, 1]
  Tinit = u_initial.(x, 0)

  #   q0 = q.(x, 0.0)
  dt = +1e-5
  Tfinal = u_analytic.(x, tfinal - dt)

  plot(xc[:, 1], T[:, 1]; label="simulation")
  plot!(x, Tfinal; label="analytic")
end

# begin
#   plot(xc[:, 1], st[:, 1]; label="simulation", xlabel="x", ylabel="q")
#   plot!(x, q.(x, 0); label="analytic")
# end
