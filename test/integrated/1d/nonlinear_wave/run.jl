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

function initialize_mesh()
  ni = 100
  nhalo = 1
  x0, x1 = (0.0, 1.0)
  return RectlinearGrid(x0, x1, ni, nhalo)
end

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
function init_state()
  mesh = adapt(ArrayT, initialize_mesh())

  # bcs = (
  #   ilo=DirichletBC(10.0),  #
  #   ihi=DirichletBC(0.0),  #
  # )
  bcs = (
    ilo=NeumannBC(),  #
    ihi=NeumannBC(),  #
  )
  solver = PseudoTransientSolver(mesh, bcs; backend=backend, face_diffusivity=:arithmetic)

  # Temperature and density
  T = zeros(Float64, cellsize_withhalo(mesh))
  T[1:2, :] .= 10
  ρ = ones(Float64, cellsize_withhalo(mesh))
  cₚ = 1.0

  fwhm = 0.01
  x0 = 0.5
  xc = Array(mesh.centroid_coordinates.x)
  for idx in mesh.iterators.cell.domain
    T[idx] = exp(-(((x0 - xc[idx])^2) / fwhm))#+ T_cold
  end

  # Define the conductivity model
  # @inline κ(ρ, T, κ0=1) = κ0 * T^3
  @inline κ(ρ, T, κ0=1) = κ0

  return solver, mesh, adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function run(maxt, maxiter=Inf)
  casename = "planar_nonlinear_heat_wave"

  scheme, mesh, T, ρ, cₚ, κ = init_state()

  x = centroids(mesh)
  domain = mesh.iterators.cell.domain

  p = plot(x, T[domain]; label="tinit")
  ylims!(0, 1.1)
  display(p)

  global Δt = 1e-4
  global t = 0.0
  global iter = 0

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

    global iter += 1
    global t += Δt
  end

  print_timer()
  return scheme, mesh, T, ρ
end

begin
  cd(@__DIR__)
  scheme, mesh, temperature, dens = run(1.0, 100)
  nothing

  x = centroids(mesh)
  domain = mesh.iterators.cell.domain

  p = plot(x, temperature[domain]; label="tfinal")
  ylims!(0, 1.1)
  display(p)

  # x = centroids(mesh)

  # domain = mesh.iterators.cell.domain
  # ddomain = scheme.iterators.domain.cartesian
  # T = @view temperature[domain]
  # st = @view scheme.source_term[ddomain]

  # front_pos = [0.870571]

  # global xfront = 0.0
  # for i in reverse(eachindex(T))
  #   if T[i] > 1e-10
  #     global xfront = x[i]
  #     break
  #   end
  # end

  # f = plot(
  #   x,
  #   T;
  #   title="Nonlinear heat front @ t = 1",
  #   label="simulation",
  #   marker=:circle,
  #   ms=2,
  #   xticks=0:0.2:1,
  #   yticks=0:0.2:1,
  # )
  # vline!(front_pos; label="analytic front position", color=:black, lw=2, ls=:dash)
  # savefig(f, "planar_nonlinear_heat_front.png")

  # f
end
