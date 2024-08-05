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
  ni = 40
  nhalo = 1
  x0, x1 = (0.0, 1.0)
  # return RectlinearGrid(x0, x1, ni, nhalo)
  # return CurvilinearGrids.GridTypes.RectlinearCylindricalGrid(
  #   x0, x1, ni, nhalo; snap_to_axis=true
  # )
  return CurvilinearGrids.GridTypes.RectlinearSphericalGrid(
    x0, x1, ni, nhalo; snap_to_axis=true
  )
end

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
function init_state()
  mesh = initialize_mesh()

  bcs = (
    # ihi=DirichletBC(1.0),  #
    ilo=NeumannBC(),  #
    ihi=NeumannBC(),  #
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

  r1 = 0.98CurvilinearGrids.radius(mesh, (mesh.nhalo + 3,))
  dep_vol = (4pi / 3 * r1^3)
  T0 = Q0 / dep_vol
  T[begin:(mesh.nhalo + 2)] .= T0

  # Define the conductivity model
  @inline κ(ρ, T) = T^n_κ

  return solver, mesh, adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function run(maxt, maxiter=Inf)
  scheme, mesh, T, ρ, cₚ, κ = init_state()

  global Δt = 1e-15
  global t = 0.0
  global iter = 0

  while true
    if iter == 0
      reset_timer!()
    end

    if iter >= maxiter || t >= maxt
      break
    end

    stats, next_Δt = nonlinear_thermal_conduction_step!(
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

    @printf "cycle: %i t: %.4e, Δt: %.3e\n" iter t Δt

    global iter += 1
    global t += Δt
    global Δt = next_Δt
  end

  print_timer()
  return scheme, mesh, T, ρ
end

begin
  cd(@__DIR__)
  tfinal = 0.3
  scheme, mesh, temperature, dens = run(tfinal, Inf)
  nothing

  x = centroids(mesh)
  domain = mesh.iterators.cell.domain

  T_ref, rf = analytical_sol(tfinal, x, n_κ)
  T_sim = temperature[domain]
  L2_error = norm((T_sim .- T_ref) / sqrt(length(T_sim)))

  p = plot(
    x,
    T_sim;
    label="Simulated",
    marker=:circle,
    xlabel="Radius",
    ylabel="T",
    title="Spherical Thermal Wave",
  )
  plot!(p, x, T_ref; label="Analytic")
  vline!([rf]; color=:black, ls=:dash, label="Wave Front")

  display(p)

  nothing
end
