using Plots
using CurvilinearGrids, CurvilinearDiffusion
using Printf, Adapt
using TimerOutputs
using KernelAbstractions
using Glob
using LinearAlgebra

dev = :CPU

const k = 1 / 30
const a = (1 + 8k) / (6 * (1 + 4k))
const b = (1 + 8k) / (12k * (1 + 4k))
const c = -1 / (12k)
T_init(x, y) = a + b * x + c * x^4
Q(x, y) = x^2

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
  ni, nj = (101, 51)
  nhalo = 2
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
    ilo=DirichletBC(a),          #
    ihi=DirichletBC(a + b + c),  #
    jlo=NeumannBC(),  #
    jhi=NeumannBC(),  #
  )
  solver = ImplicitScheme(mesh, bcs; backend=backend, direct_solve=true)

  # Temperature and density
  T = zeros(Float64, cellsize_withhalo(mesh))
  source_term = zeros(Float64, cellsize_withhalo(mesh))
  ρ = ones(Float64, cellsize_withhalo(mesh))
  cₚ = 1.0

  # Define the conductivity model
  @inline κ(ρ, temperature) = 1.0

  for idx in mesh.iterators.cell.domain
    x⃗c = centroid(mesh, idx)
    # T[idx] = T_init(x⃗c...)
    source_term[idx] = Q(x⃗c...)
  end

  scheme_q = @view solver.source_term[solver.iterators.domain.cartesian]
  mesh_q = @view source_term[mesh.iterators.cell.domain]
  copy!(scheme_q, mesh_q)

  #   fill!(solver.source_term, 0.0)
  fill!(solver.α, k)

  return solver, mesh, adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function run(maxt, maxiter=Inf)
  casename = "steady_state_nonlinear"

  scheme, mesh, T, ρ, cₚ, κ = init_state()
  global Δt = 5e-3
  global t = 0.0
  global iter = 0
  global io_interval = 0.1
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

    # if t + Δt > io_next
    #   @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)
    #   global io_next += io_interval
    # end

    global iter += 1
    global t += Δt
  end

  # @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)

  print_timer()
  return scheme, mesh, T
end

begin
  cd(@__DIR__)
  rm.(glob("*.vts"))
  tfinal = 1.0
  scheme, mesh, temperature = run(tfinal, Inf)
  nothing

  xc, yc = centroids(mesh)

  domain = mesh.iterators.cell.domain
  ddomain = scheme.iterators.domain.cartesian
  T = @view temperature[domain]
  st = @view scheme.source_term[ddomain]

  x = xc[:, 1]
  Tfinal = T_init.(x, nothing)

  plot(x, T[:, 1]; label="simulation")
  plot!(x, Tfinal; label="analytic")
end
