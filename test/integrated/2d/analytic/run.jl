using CairoMakie
using CurvilinearGrids, CurvilinearDiffusion
using Printf, Adapt
using TimerOutputs
using KernelAbstractions
using Glob
using LinearAlgebra

const π² = π * π
const q0 = 2

function u_initial(x, t)
  return sinpi(3x)
end

function q(x, t)
  return 0 #q0 * sinpi(x)
end

function u_analytic(x, t)
  return sinpi(3x) * exp(-9π² * t) #+ (q0 / π²) * sinpi(x) * (1 - exp(-π² * t))
end

# u_initial(x, t) = 0.25sinpi(2x) + sinpi(3x)
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
function wavy_grid(nx, ny, nhalo)
  x1d = range(0, 1; length=nx)
  y1d = range(0, 1; length=ny)

  dx = first(diff(x1d))
  dy = first(diff(y1d))
  # dx = 1 / (nx - 1)
  # dy = 1 / (ny - 1)
  # x1d = (0.5dx):dx:(1 - 0.5dx)
  # y1d = (0.5dy):dy:(1 - 0.5dy)
  x2d = zeros(length(x1d), length(y1d))
  y2d = zeros(length(x1d), length(y1d))

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

  # scale such that the ghost layer is at 0, so the boundary
  # conditions are correct...
  @. x2d = (x2d * (1 - dx)) + 0.5dx
  @. y2d = (y2d * (1 - dy)) + 0.5dy
  return CurvilinearGrids.CurvilinearGrid2D(x2d, y2d, nhalo)
end

function uniform_grid(nx, ny, nhalo)
  x0, x1 = (0, 1)
  y0, y1 = (0, 1)

  # dx = 1 / (nx - 1)
  # dy = 1 / (ny - 1)
  # x = (0.5dx):dx:(1 - 0.5dx) |> collect
  # y = (0.5dy):dy:(1 - 0.5dy) |> collect

  # return CurvilinearGrids.RectlinearGrid(x, y, nhalo, CPU())
  return CurvilinearGrids.RectlinearGrid((x0, y0), (x1, y1), (nx, ny), nhalo, CPU())
end

function initialize_mesh()
  ni = nj = 451
  nhalo = 1
  # return wavy_grid(ni, nj, nhalo)
  return uniform_grid(ni, nj, nhalo)
end

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------

# Define the conductivity model
@inline κ(ρ, temperature) = 1.0

function init_state()
  mesh = adapt(ArrayT, initialize_mesh())

  bcs = (
    # ilo=DirichletBC((; ρ=1.0, u=0, α=1.0)),  #
    # ihi=DirichletBC((; ρ=1.0, u=0, α=1.0)),  #
    ilo=FixedNegSymmetryBC(),  #
    ihi=FixedNegSymmetryBC(),  #
    # jlo=NeumannBC(),  #
    # jhi=NeumannBC(),  #
    jlo=PeriodicBC(),  #
    jhi=PeriodicBC(),  #
  )

  # solver = ImplicitScheme(mesh, bcs; backend=backend)
  solver = PseudoTransientSolver(mesh, bcs; backend=backend, mean=:harmonic)

  # Temperature and density
  T = zeros(Float64, cellsize_withhalo(mesh))
  source_term = zeros(Float64, cellsize_withhalo(mesh))
  ρ = ones(Float64, cellsize_withhalo(mesh))
  cₚ = 1.0

  xc = Array(mesh.centroid_coordinates.x)
  # yc = Array(mesh.centroid_coordinates.y)
  for idx in mesh.iterators.cell.domain
    # x⃗c = centroid(mesh, idx)
    T[idx] = u_initial(xc[idx], 0.0)
    source_term[idx] = q(xc[idx], 0.0) # / mesh.cell_center_metrics.J[idx]
  end

  scheme_q = @view solver.source_term[solver.iterators.domain.cartesian]
  mesh_q = source_term[mesh.iterators.cell.domain]
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
  global Δt = 1e-7
  global t = 0.0
  global iter = 0
  global io_interval = 0.01
  global io_next = io_interval
  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, ρ, mesh, iter, t, casename)

  while true
    if iter == 0
      reset_timer!()
    end

    @printf "cycle: %i t: %.4e, Δt: %.3e\n" iter t Δt

    stats, next_dt = nonlinear_thermal_conduction_step!(
      scheme,
      mesh,
      T,
      ρ,
      cₚ,
      κ,
      Δt;
      cutoff=false,
      rel_tol=1e-9,
      abs_tol=1e-8,
      enforce_positivity=false,
      subcycle_conductivity=false,
    )

    if t + Δt > io_next
      @timeit "save_vtk" CurvilinearDiffusion.save_vtk(
        scheme, T, ρ, mesh, iter, t, casename
      )
      global io_next += io_interval
    end

    if iter >= maxiter || t >= maxt
      break
    end

    global iter += 1
    global t += Δt

    if isfinite(next_dt)
      Δt = min(next_dt, 1e-4)
    end
  end

  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, ρ, mesh, iter, t, casename)

  print_timer()
  return scheme, mesh, T
end

begin
  cd(@__DIR__)
  rm.(glob("*.vts"))
  # tfinal = 0.03
  tfinal = 0.04
  scheme, mesh, temperature = run(tfinal, Inf)
  nothing
end

begin
  xc, yc = centroids(mesh)

  domain = mesh.iterators.cell.domain
  ddomain = scheme.iterators.domain.cartesian
  sim = @view temperature[domain]

  # tfinal = 4.9683e-02
  # tfinal = 0.0505
  sol = u_analytic.(xc, tfinal)
  x = xc[:, 1]
  Tinit = u_initial.(x, 0)

  L₂ = sqrt(mapreduce((x, y) -> abs(x^2 - y^2), +, sol, sim) / length(sim))

  @show L₂
  f = Figure(; size=(800, 400))
  ax1 = Axis(
    f[1, 1];
    aspect=1,
    xlabel="x",
    ylabel="u",
    xgridvisible=false,
    ygridvisible=false,
    xticks=-1:0.5:1,
    yticks=-1:0.5:1,
  )
  ax2 = Axis(
    f[1, 2]; aspect=1, xlabel="x", ylabel="u", xgridvisible=false, ygridvisible=false
  )

  lines!(ax1, xc[:, 1], Tinit; label="initial")
  axislegend(ax1; position=:lb)

  # lines!(ax2, xc[:, 1], Tsim; label="t_final")
  lines!(ax2, xc[:, 1], sim[:, 1]; color=:red, label="simulated")
  scatter!(ax2, vec(xc), vec(sol); markersize=4, color=:black, label="analytic")

  # lines!(ax2, x, sol; label="analytic", linestyle=:dash, linewidth=1)
  # axislegend(; position=:cb)
  axislegend(;)
  save("$(@__DIR__)/multimode_sine.png", f)
  display(f)
end

# L₂ = 0.0028167836012538065 # 50

begin
  res = 1 ./ [50, 150, 450]
  err = [
    0.0028167836012538065, # 50
    0.000743386916876679, # 150
    0.0005328113359953201, # 450
  ]

  f = Figure()
  ax = Axis(f[1, 1]; xscale=log10, yscale=log10)
  scatterlines!(ax, res, err)
  scatterlines!(ax, res, res .^ 1.2)
  f
end

q_2 = log(err[3] / err[2]) / log(res[3] / res[2])
q_2 = log(err[2] / err[1]) / log(res[2] / res[1])
q_2 = log(err[1] / err[2]) / log(res[1] / res[2])
err[2] / err[1]
res[2] / res[1]

# sim[:,1] |> lines
# sol[:,1] |> lines

# delta_sol = sim .- sol
# delta_sol[:, 1] |> lines

# heatmap(sim .- sol)

# extrema(sim .- sol)
# lines(vec(abs.(sim .- sol)))
# f, ax, p = heatmap(sim - sol)
# Colorbar(f, p)
# f

# # begin
# resolution = 1 ./ [50, 100, 200]

# # wavy mesh error
# err = [
#   0.010542925663995471,  # 50
#   0.007400455795832716,  # 100
#   0.0051917456471774134, # 200
# ]

# # uniform mesh error
# err = [
#   0.00440401873833551,  # 50
#   0.00331829403959398,  # 100 with fixed Δt; 0.00903241726146163 with variable Δt
#   0.0, # 200
# ]

# #   # res_order = 1 ./ [50, 100]
# #   err_order = @. resolution .^ 2

# #   f = Figure(; size=(400, 400))
# #   ax = Axis(
# #     f[1, 1];
# #     aspect=1,
# #     xlabel="dx",
# #     yscale=log10,
# #     xscale=log10,
# #     ylabel="L₂ error",
# #     xgridstyle=:dash,
# #     ygridstyle=:dash,
# #     yminorticksvisible=true,
# #     # yticks=[1e-3, 1e-1],
# #     yminorticks=IntervalsBetween(9),

# #     # xgridvisible=false,
# #     # ygridvisible=false,
# #   )

# #   scatterlines!(ax, resolution, err)
# #   lines!(ax, resolution, err_order)
# #   # ylims!(1e-3, 1e-1)
# #   # xlims!(1e1, 1e3)

# #   # axislegend(ax; position=:lb)

# #   #   # save("$(@__DIR__)/multimode_sine.png", f)
# #   f
# # end
