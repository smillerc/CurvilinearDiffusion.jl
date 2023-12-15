using CurvilinearGrids, CurvilinearDiffusion
using WriteVTK, Printf

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
  ilo, ihi, jlo, jhi = mesh.limits

  α = @views solver.diffusivity[ilo:ihi, jlo:jhi]
  dens = @views ρ[ilo:ihi, jlo:jhi]
  temp = @views T[ilo:ihi, jlo:jhi]

  @show size(α), size(dens), size(temp)
  kappa = α .* dens

  xy_n = CurvilinearGrids.coords(mesh)
  vtk_grid(fn, xy_n) do vtk
    vtk["TimeValue"] = t
    vtk["density"] = dens
    vtk["temperature"] = temp
    vtk["diffusivity"] = α
    vtk["conductivity"] = kappa
    pvd[t] = vtk
  end
end

# ------------------------------------------------------------
# Grid Construction
# ------------------------------------------------------------

function wavy_grid(nx, ny)
  x0, x1 = (0, 1)
  y0, y1 = (0, 1)
  a0 = 0.1

  x1d(i) = @. x0 + (x1 - x0) * ((i - 1) / (nx - 1))
  y1d(j) = @. y0 + (y1 - y0) * ((j - 1) / (ny - 1))

  x(i, j) = @. x1d(i) + a0 * sinpi(2x1d(i)) * sinpi(2y1d(j))
  y(i, j) = @. y1d(j) + a0 * sinpi(2x1d(i)) * sinpi(2y1d(j))

  return (x, y)
end

function uniform_grid(nx, ny)
  x0, x1 = (0, 1)
  y0, y1 = (0, 1)

  x(i, j) = @. x0 + (x1 - x0) * ((i - 1) / (nx - 1))
  y(i, j) = @. y0 + (y1 - y0) * ((j - 1) / (ny - 1))

  return (x, y)
end

ni, nj = (41, 41)
nhalo = 1
# x, y = wavy_grid(ni, nj)
x, y = uniform_grid(ni, nj)
mesh = CurvilinearGrid2D(x, y, (ni, nj), nhalo)

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
T0 = 1.0
bcs = (
  ilo=(:fixed, T0),
  ihi=(:fixed, 0.0),
  # jlo=:zero_flux,
  # jhi=:zero_flux,
  jlo=:periodic,
  jhi=:periodic,
)

solver = ADESolver(mesh, bcs)
CFL = 1 # 1/2 is the explicit stability limit
casename = "nonlinear_coldwall"

# Temperature and density
T = ones(Float64, cellsize_withhalo(mesh)) * 1e-1
ρ = ones(Float64, cellsize_withhalo(mesh))
cₚ = 1.0

# ilo = mesh.limits.ilo
# T[begin:(ilo - 1), :] .= Tbc

# Define the conductivity model
@inline function κ(ρ, T)
  if !isfinite(T)
    return 0.0
  else
    κ0 = 1.0
    return κ0 * T^3
  end
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------

Δt = 1e-5
t = 0.0
maxt = 1.0
iter = 1
maxiter = Inf
io_interval = 0.01
io_next = io_interval
pvd = paraview_collection("full_sim")
save_vtk(solver, ρ, T, mesh, iter, t, casename, pvd)
while true
  CurvilinearDiffusion.update_conductivity!(solver, T, ρ, κ, cₚ)
  # dt = CurvilinearDiffusion.max_dt(solver, mesh)
  # global Δt = CFL * dt

  L₂, Linf = CurvilinearDiffusion.solve!(solver, T, Δt)
  @printf "cycle: %i t: %.4e, L2: %.1e, L∞: %.1e Δt: %.3e\n" iter t L₂ Linf Δt

  # if iter % io_interval == 0
  if t + Δt > io_next
    save_vtk(solver, ρ, T, mesh, iter, t, casename, pvd)
    global io_next += io_interval
  end

  if iter >= maxiter
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