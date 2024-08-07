using Plots
using CurvilinearGrids, CurvilinearDiffusion
using Printf, Adapt
using TimerOutputs
using KernelAbstractions
using Glob
using LinearAlgebra
using SpecialFunctions

const n_κ = 2 # exponent used in thermal conductivity

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
# Analytical Solution
# ------------------------------------------------------------
"""Dimensionless constant -- used in the analytical solution"""
function ξ(n)
  return (
    ((3n + 2) / (2^(n - 1) * n * π^n))^(1 / (3n + 2)) *
    (gamma(5 / 2 + 1 / n) / (gamma(1 + 1 / n) * gamma(3 / 2)))^(n / (3n + 2))
  )
end

"""Analytical solution for the central temperature"""
function central_temperature(t, n, Q0, ρ, cᵥ, κ0)
  ξ₁ = ξ(n)

  Tc = (
    ((n * ξ₁^2) / (2 * (3n + 2)))^(1 / n) *
    Q0^(2 / (3n + 2)) *
    ((ρ * cᵥ) / (κ0 * t))^(3 / (3n + 2))
  )
  return Tc
end

"""Analytical solution for the wave front radius"""
function wave_front_radius(t, n, Q0, ρ, cᵥ, κ0)
  ξ₁ = ξ(n)

  return ξ₁ * ((κ0 * t * Q0^n) / (ρ * cᵥ))^(1 / (3n + 2))
end

"""Analytical solution of T(r,t)"""
function analytical_sol(t, r, n, Q0=1, ρ=1, cᵥ=1, κ0=1)
  T = zeros(size(r))
  Tc = central_temperature(t, n, Q0, ρ, cᵥ, κ0)
  rf = wave_front_radius(t, n, Q0, ρ, cᵥ, κ0)
  rf² = rf^2

  for i in eachindex(T)
    rterm = (1 - (r[i]^2 / rf²))
    if rterm > 0
      T[i] = Tc * rterm^(1 / n)
    end
  end

  return T, rf
end

# ------------------------------------------------------------
# Grid Construction
# ------------------------------------------------------------

function initialize_mesh()
  ni, nj = (50, 150)
  nhalo = 1
  x0, x1 = (0.0, 1.0)
  y0, y1 = (0.0, 1.0)

  # return CurvilinearGrids.GridTypes.AxisymmetricRectlinearGrid(
  #   (x0, y0), (x1, y1), (ni, nj), nhalo; snap_to_axis=true, rotational_axis=:x
  # )

  r = range(1e-9, 1; length=51)
  θ = range(0, pi / 2; length=51)
  return AxisymmetricRThetaGrid(r, θ, nhalo; snap_to_axis=true, rotational_axis=:x)

  # return CurvilinearGrids.GridTypes.RectlinearGrid((x0, y0), (x1, y1), (ni, nj), nhalo)
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

  solver = PseudoTransientSolver(
    mesh,
    bcs; #
    backend=backend, #
    face_diffusivity=:arithmetic,
  )

  # Temperature and density
  T = ones(Float64, cellsize_withhalo(mesh)) * 1e-6
  ρ = ones(Float64, cellsize_withhalo(mesh))
  cₚ = 1.0
  Q0 = 1.0

  # T0 = Q0 / 2cellvolume(mesh, (2, 2))
  T0 = 1100.0
  @show T0
  # error("done")
  # T[begin:(mesh.nhalo + 1), begin:(mesh.nhalo + 1)] .= T0
  T[begin:(mesh.nhalo + 1), :] .= T0
  # T[25, :] .= T0
  copy!(solver.u, T)

  # Define the conductivity model
  @inline κ(ρ, T) = T^2

  return solver, mesh, adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function run(maxt, maxiter=Inf)
  casename = "axisymmetric"
  scheme, mesh, T, ρ, cₚ, κ = init_state()

  global Δt = 1e-15
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

    if t + Δt > maxt
      Δt = maxt - t
    end

    @printf "cycle: %i t: %.4e, Δt: %.3e\n" iter t Δt
    _, next_Δt = nonlinear_thermal_conduction_step!(
      scheme,
      mesh,
      T,
      ρ,
      cₚ,
      κ,
      Δt;
      cutoff=false,
      apply_density_bc=false,
      max_iter=10000,
      # abs_tol=5e-5,
    )

    # if t + Δt > io_next
    if iter % 50 == 0
      @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)
      #   global io_next += io_interval
    end

    global iter += 1
    global t += Δt
    global Δt = next_Δt
  end

  @printf "Final t: %.4e\n" t

  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)

  print_timer()
  return scheme, mesh, T, ρ
end

begin
  cd(@__DIR__)
  rm.(glob("*.vts"))
  tfinal = 0.3
  scheme, mesh, temperature, dens = run(tfinal, 5000)

  # xc, yc = centroids(mesh)
  # x = xc[:, 1]
  # domain = mesh.iterators.cell.domain

  # T_ref, rf = analytical_sol(tfinal, x, n_κ)
  # T_sim = temperature[domain]
  # L2_error = norm((T_sim .- T_ref) / sqrt(length(T_sim)))

  # p = plot(
  #   x,
  #   T_sim[:, 1];
  #   label="Simulated",
  #   marker=:circle,
  #   xlabel="Radius",
  #   ylabel="T",
  #   title="Spherical Thermal Wave",
  # )
  # plot!(p, x, T_ref; label="Analytic")
  # vline!([rf]; color=:black, ls=:dash, label="Wave Front")

  # display(p)

  nothing
end

begin
  T_sim = temperature[domain]
  xc, yc = centroids(mesh)
  x = xc[:, 1]
  domain = mesh.iterators.cell.domain

  volumes = CurvilinearGrids.GridTypes.cellvolumes(mesh)
  v_central = volumes[1, :]

  Q0 = mapreduce((v, t) -> v * t, +, v_central, T_sim[1, :]) * 50000

  T_ref, rf = analytical_sol(tfinal, x, n_κ)
  L2_error = norm((T_sim .- T_ref) / sqrt(length(T_sim)))

  p = plot(
    x,
    T_sim[:, 1];
    label="Simulated",
    marker=:circle,
    xlabel="Radius",
    ylabel="T",
    title="Spherical Thermal Wave",
  )
  plot!(p, x, T_ref; label="Analytic")
  vline!([rf]; color=:black, ls=:dash, label="Wave Front")

  display(p)
end
