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
function wavy_grid(ni, nj, nk, nhalo)
  Lx = Ly = Lz = 12

  xmin = -Lx / 2
  ymin = -Ly / 2
  zmin = -Lz / 2

  Δx0 = Lx / ni
  Δy0 = Ly / nj
  Δz0 = Lz / nk

  Ax = 0.2 / Δx0
  Ay = 0.2 / Δy0
  Az = 0.2 / Δz0

  x = zeros(ni, nj, nk)
  y = zeros(ni, nj, nk)
  z = zeros(ni, nj, nk)

  n = 0.5
  for k in 1:nk
    for j in 1:nj
      for i in 1:ni
        x[i, j, k] =
          xmin + Δx0 * ((i - 1) + Ax * sinpi(n * (j - 1) * Δy0) * sinpi(n * (k - 1) * Δz0))
        y[i, j, k] =
          ymin + Δy0 * ((j - 1) + Ay * sinpi(n * (k - 1) * Δz0) * sinpi(n * (i - 1) * Δx0))
        z[i, j, k] =
          zmin + Δz0 * ((k - 1) + Az * sinpi(n * (i - 1) * Δx0) * sinpi(n * (j - 1) * Δy0))
      end
    end
  end

  return CurvilinearGrid3D(x, y, z, nhalo)
end

function uniform_grid(nx, ny, nz, nhalo)
  x0, x1 = (-6, 6)
  y0, y1 = (-6, 6)
  z0, z1 = (-6, 6)

  return CurvilinearGrids.RectlinearGrid((x0, y0, z0), (x1, y1, z1), (nx, ny, nz), nhalo)
end

function initialize_mesh()
  ni, nj, nk = (50, 50, 50)
  nhalo = 4
  return uniform_grid(ni, nj, nk, nhalo)
  # return wavy_grid(ni, nj, nk, nhalo)
end

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
function init_state()
  # mesh = adapt(ArrayT, initialize_mesh())
  mesh = initialize_mesh()

  bcs = (
    ilo=NeumannBC(),  #
    ihi=NeumannBC(),  #
    # ilo=DirichletBC(1.0),  #
    # ihi=DirichletBC(0.0),  #
    jlo=NeumannBC(),  #
    jhi=NeumannBC(),  #
    klo=NeumannBC(),  #
    khi=NeumannBC(),  #
  )

  solver = PseudoTransientSolver(mesh, bcs; backend=backend)

  # Temperature and density
  T_hot = 1e3
  T_cold = 1e-2
  T = ones(size(mesh.iterators.cell.full)) * T_cold
  ρ = ones(size(mesh.iterators.cell.full))
  source_term = ones(size(mesh.iterators.cell.full))
  cₚ = 1.0

  # Define the conductivity model
  @inline function κ(ρ, temperature, κ0=1.0)
    if !isfinite(temperature)
      return 0.0
    else
      return κ0 # * temperature^3
    end
  end

  fwhm = 1.0
  x0 = y0 = z0 = 0.0

  xc = Array(mesh.centroid_coordinates.x)
  yc = Array(mesh.centroid_coordinates.y)
  zc = Array(mesh.centroid_coordinates.z)
  for idx in mesh.iterators.cell.domain
    T[idx] =
      T_hot * exp(
        -(((x0 - xc[idx])^2) / fwhm + ((y0 - yc[idx])^2) / fwhm + ((z0 - zc[idx])^2) / fwhm)
      ) # + T_cold
  end

  # copy!(solver.source_term, source_term) # move to gpu (if need be)

  return solver, mesh, adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function run(maxiter=Inf)
  casename = "pt_uniform_3d_no_source"

  scheme, mesh, T, ρ, cₚ, κ = init_state()

  global Δt = 1e-4
  #   global Δt = 0.02
  global t = 0.0
  global maxt = 0.6

  global iter = 0
  global io_interval = 0.01
  global io_next = io_interval

  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)

  while true
    if iter == 0
      reset_timer!()
    end

    global iter += 1
    global t += Δt

    @printf "cycle: %i t: %.4e, Δt: %.3e\n" iter t Δt
    err, subiter = CurvilinearDiffusion.PseudoTransientScheme.step!(
      scheme, mesh, T, ρ, cₚ, κ, Δt; max_iter=15, tol=5e-8, error_check_interval=2
    )
    @printf "\tL2: %.3e, %i\n" err subiter

    if t + Δt > io_next
      @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, mesh, iter, t, casename)
      global io_next += io_interval
    end

    if t >= maxt
      break
    end

    if iter >= maxiter - 1
      break
    end
    # Δt = next_dt
  end

  @timeit "save_vtk" CurvilinearDiffusion.save_vtk(
    scheme, scheme.u, mesh, iter, t, casename
  )

  print_timer()
  return scheme, mesh, T
end

begin
  cd(@__DIR__)
  rm.(glob("*.vts"))

  # NVTX.@range "my message" begin
  scheme, mesh, temperature = run(100)
  nothing
  # end
end
