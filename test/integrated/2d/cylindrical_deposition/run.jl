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
function initialize_mesh()
  nhalo = 2

  r0, r1 = @. (10.0, 150.0) # min/max radius
  (θ0, θ1) = deg2rad.((20, 90)) ./ pi # min/max polar angle

  nr, ntheta = 51, 51

  # Linear spacing in each dimension
  r(ξ) = r0 + (r1 - r0) * ((ξ - 1) / (nr - 1))
  θ(η) = θ0 + (θ1 - θ0) * ((η - 1) / (ntheta - 1))

  R(i, j) = r(i) * cospi(θ(j))
  Z(i, j) = r(i) * sinpi(θ(j))

  # mesh = CylindricalGrid2D(R, Z, (nr, ntheta), nhalo)
  mesh = CurvilinearGrid2D(R, Z, (nr, ntheta), nhalo)
  return mesh
end

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
function init_state()
  mesh = initialize_mesh()
  bcs = (
    ilo=NeumannBC(),  #
    ihi=NeumannBC(),  #
    jlo=NeumannBC(),  #
    jhi=NeumannBC(),  #
  )

  backend = CPU()
  # solver = ImplicitScheme(mesh, bcs; backend=backend, direct_solve=true)
  solver = ADESolver(mesh, bcs; backend=backend, face_conductivity=:arithmetic)

  # Temperature and density
  T_hot = 1e2
  T_cold = 1e-2
  T = ones(Float64, cellsize_withhalo(mesh)) * T_cold
  ρ = ones(Float64, cellsize_withhalo(mesh))
  source_term = similar(solver.source_term)
  fill!(source_term, 0)
  cₚ = 1.0

  # Define the conductivity model
  @inline function κ(ρ, temperature, κ0=10.0)
    if !isfinite(temperature)
      return 0.0
    else
      return κ0 * temperature^3
    end
  end

  i_dep = 25:30
  @views begin
    source_term[i_dep, :] .= T_hot
  end

  copy!(solver.source_term, source_term) # move to gpu (if need be)

  return solver, mesh, T, ρ, cₚ, κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function run(maxiter=Inf)
  casename = "cylindrical_deposition"

  scheme, mesh, T, ρ, cₚ, κ = init_state()

  global Δt = 1e-4
  global t = 0.0
  global maxt = 1.0
  global iter = 0
  global io_interval = 0.01
  global io_next = io_interval
  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)

  while true
    if iter == 0
      reset_timer!()
    end

    nonlinear_thermal_conduction_step!(scheme, mesh, T, ρ, cₚ, κ, Δt)

    @printf "cycle: %i t: %.4e, Δt: %.3e\n" iter t Δt

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
  end

  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)

  print_timer()
  return scheme, mesh, T
end

begin
  cd(@__DIR__)
  rm.(glob("*.vts"))

  solver, mesh, temperature = run(10)
  nothing
end

# i_bc = (
#   halo=(lo=CartesianIndices((3:3, 4:51)), hi=CartesianIndices((52:52, 4:51))),
#   edge=(lo=CartesianIndices((4:4, 4:51)), hi=CartesianIndices((51:51, 4:51))),
# )
# j_bc = (
#   halo=(lo=CartesianIndices((4:51, 3:3)), hi=CartesianIndices((4:51, 52:52))),
#   edge=(lo=CartesianIndices((4:51, 4:4)), hi=CartesianIndices((4:51, 51:51))),
# )

# i_bc = (
#   halo=(lo=CartesianIndices((1:1, 2:53)), hi=CartesianIndices((54:54, 2:53))),
#   edge=(lo=CartesianIndices((2:2, 2:53)), hi=CartesianIndices((53:53, 2:53))),
# )
# j_bc = (
#   halo=(lo=CartesianIndices((2:53, 1:1)), hi=CartesianIndices((2:53, 54:54))),
#   edge=(lo=CartesianIndices((2:53, 2:2)), hi=CartesianIndices((2:53, 53:53))),
# )