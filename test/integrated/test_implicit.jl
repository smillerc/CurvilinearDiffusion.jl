using CurvilinearGrids, CurvilinearDiffusion
using WriteVTK, Printf, UnPack, Adapt
using BlockHaloArrays
using TimerOutputs
using KernelAbstractions
using Glob
using LinearAlgebra

@static if Sys.islinux()
  using MKL
elseif Sys.isapple()
  using AppleAccelerate
end

NMAX = Sys.CPU_THREADS
BLAS.set_num_threads(NMAX)
BLAS.get_num_threads()

@show BLAS.get_config()

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
# I/O stuff
# ------------------------------------------------------------

get_filename(iteration) = file_prefix * @sprintf("%07i", iteration)
function get_filename(iteration, name::String)
  name = replace(name, " " => "_")
  if !endswith(name, "_")
    name = name * "_"
  end
  return name * @sprintf("%07i", iteration)
end

function save_vtk(solver, ρ, T, mesh, iteration, t, name, pvd)
  fn = get_filename(iteration, name)
  @info "Writing to $fn"
  ilo, ihi, jlo, jhi = mesh.domain_limits.cell

  α = @views solver.α[ilo:ihi, jlo:jhi]
  dens = @views ρ[ilo:ihi, jlo:jhi]
  temp = @views T[ilo:ihi, jlo:jhi]
  block = zeros(Int, size(dens))
  q = @views solver.source_term[ilo:ihi, jlo:jhi]
  kappa = α .* dens

  # ξx = [m.ξx for m in solver.metrics[ilo:ihi, jlo:jhi]]
  # ξxim1 = [m.ξxᵢ₋½ for m in solver.metrics[ilo:ihi, jlo:jhi]]
  # ξxip1 = [m.ξxᵢ₊½ for m in solver.metrics[ilo:ihi, jlo:jhi]]
  # ξy = [m.ξy for m in solver.metrics[ilo:ihi, jlo:jhi]]
  # ηx = [m.ηx for m in solver.metrics[ilo:ihi, jlo:jhi]]
  # ηy = [m.ηy for m in solver.metrics[ilo:ihi, jlo:jhi]]
  # J = [m.J for m in solver.metrics[ilo:ihi, jlo:jhi]]

  xy_n = CurvilinearGrids.coords(mesh)
  vtk_grid(fn, xy_n) do vtk
    vtk["TimeValue"] = t
    vtk["density"] = dens
    vtk["temperature"] = temp
    # vtk["diffusivity"] = α
    vtk["conductivity"] = kappa
    vtk["block"] = block
    vtk["source_term"] = q

    # vtk["xi_xim1"] = ξxim1
    # vtk["xi_xip1"] = ξxip1
    # vtk["xi_x"] = ξx
    # vtk["xi_y"] = ξy
    # vtk["eta_x"] = ηx
    # vtk["eta_y"] = ηy
    # vtk["J"] = @view solver.J[ilo:ihi, jlo:jhi]

    pvd[t] = vtk
  end
end

# ------------------------------------------------------------
# Grid Construction
# ------------------------------------------------------------
function wavy_grid(nx, ny)
  x0, x1 = (-6, 6)
  y0, y1 = (-6, 6)
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
  x0, x1 = (-6, 6)
  y0, y1 = (-6, 6)

  x(i, j) = @. x0 + (x1 - x0) * ((i - 1) / (nx - 1))
  y(i, j) = @. y0 + (y1 - y0) * ((j - 1) / (ny - 1))

  return (x, y)
end

function initialize_mesh()
  ni, nj = (101, 101)
  nhalo = 6
  x, y = wavy_grid2(ni, nj)
  # x, y = uniform_grid(ni, nj)
  return CurvilinearGrid2D(x, y, (ni, nj), nhalo)
end

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
function init_state()
  mesh = adapt(ArrayT, initialize_mesh())

  T0 = 1.0
  bcs = (ilo=:zero_flux, ihi=:zero_flux, jlo=:zero_flux, jhi=:zero_flux)

  solver = ImplicitScheme(mesh, bcs; backend=backend, direct_solve=true)
  CFL = 100 # 1/2 is the explicit stability limit

  # Temperature and density
  T_hot = 1e2
  T_cold = 1e-2
  T = ones(Float64, cellsize_withhalo(mesh)) * T_cold
  ρ = ones(Float64, cellsize_withhalo(mesh))
  source_term = similar(ρ)
  cₚ = 1.0

  # Define the conductivity model
  @inline function κ(ρ, temperature, κ0=10.0)
    if !isfinite(temperature)
      return 0.0
    else
      return κ0 #* temperature^3
    end
  end

  # Gaussian source term
  fwhm = 0.5
  x0 = 0.0
  y0 = 0.0
  for idx in mesh.iterators.cell.domain
    x⃗c = centroid(mesh, idx)

    # source_term[idx] =
    #   T_hot * exp(-(((x0 - x⃗c.x)^2) / fwhm + ((y0 - x⃗c.y)^2) / fwhm)) + T_cold
    T[idx] = T_hot * exp(-(((x0 - x⃗c.x)^2) / fwhm + ((y0 - x⃗c.y)^2) / fwhm)) + T_cold
  end

  fill!(source_term, 0)
  copy!(solver.source_term, source_term) # move to gpu (if need be)
  # copy!(solver.source_term, source_term) # move to gpu (if need be)

  return solver, mesh, adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function run(maxiter=Inf)
  solver, mesh, T, ρ, cₚ, κ = init_state()

  global Δt = 1e-5
  global t = 0.0
  global maxt = 0.015
  global iter = 0
  global io_interval = 5e-4
  global io_next = io_interval
  pvd = paraview_collection("full_sim")
  @timeit "save_vtk" save_vtk(solver, ρ, T, mesh, iter, t, casename, pvd)

  @timeit "update_conductivity!" CurvilinearDiffusion.update_conductivity!(
    solver.α, T, ρ, κ, cₚ
  )

  while true
    if iter == 0
      reset_timer!()
    end

    # dt = CurvilinearDiffusion.max_dt(solver, mesh)
    # dt = max(1e-10, min(Δt0, dt))
    # global Δt = CFL * dt
    # @show Δt, CFL, dt

    # @timeit "solve!" L₂, ncycles, is_converged = CurvilinearDiffusion.ImplicitSchemeType.solve!(
    #   solver, mesh, T, Δt
    # )
    @timeit "solve!" CurvilinearDiffusion.ImplicitSchemeType.solve!(solver, mesh, T, Δt)
    @printf "cycle: %i t: %.4e, Δt: %.3e\n" iter t Δt

    if t + Δt > io_next
      @timeit "save_vtk" save_vtk(solver, ρ, T, mesh, iter, t, casename, pvd)
      global io_next += io_interval
    end

    if iter >= maxiter - 1
      break
    end

    if t >= maxt
      break
    end

    global iter += 1
    global t += Δt
  end

  save_vtk(solver, ρ, T, mesh, iter, t, casename, pvd)
  vtk_save(pvd)

  print_timer()
  return solver, T
end

begin
  cd(@__DIR__)
  const casename = "implicit_gauss_source"
  rm.(glob("*.vts"))

  solver, temperature = run(1500)
  nothing
end

# T_cpu = Array(temperature)
# using Plots: heatmap

# heatmap(T_cpu)
# solver, mesh, T, ρ, cₚ, κ = init_state();
