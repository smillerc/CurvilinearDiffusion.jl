using CurvilinearGrids, CurvilinearDiffusion
using WriteVTK, Printf, UnPack
using BlockHaloArrays

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

  α = @views solver.a[ilo:ihi, jlo:jhi]
  dens = @views ρ[ilo:ihi, jlo:jhi]
  temp = @views T[ilo:ihi, jlo:jhi]
  block = zeros(Int, size(dens))

  for blk in 1:nblocks(solver.u[1])
    globalCI = CartesianIndices(solver.u[1].global_blockranges[blk])
    for idx in globalCI
      block[idx] = blk
    end
  end
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
    vtk["diffusivity"] = α
    vtk["conductivity"] = kappa
    vtk["block"] = block

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
  x0, x1 = (0, 1)
  y0, y1 = (0, 1)
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
  x0, x1 = (0, 1)
  y0, y1 = (0, 1)

  x(i, j) = @. x0 + (x1 - x0) * ((i - 1) / (nx - 1))
  y(i, j) = @. y0 + (y1 - y0) * ((j - 1) / (ny - 1))

  return (x, y)
end

ni, nj = (101, 101)
nhalo = 1
x, y = wavy_grid2(ni, nj)
# x, y = uniform_grid(ni, nj)
mesh = CurvilinearGrid2D(x, y, (ni, nj), nhalo)

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
T0 = 1.0
bcs = (ilo=:zero_flux, ihi=:zero_flux, jlo=:zero_flux, jhi=:zero_flux)

n_blocks = 4
solver = BlockADESolver(mesh, bcs, n_blocks)
CFL = 0.1 # 1/2 is the explicit stability limit
casename = "gauss_source"

# Temperature and density
T_hot = 1e3
T_cold = 1e-2
T = ones(Float64, cellsize_withhalo(mesh)) * T_cold
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

# Gaussian source term
fwhm = 0.5
x0 = 0.0
y0 = 0.0
@unpack ilo, ihi, jlo, jhi = mesh.limits
for j in jlo:jhi
  for i in ilo:ihi
    c_loc = centroid(mesh, (i, j))

    solver.source_term[i, j] =
      T_hot * exp(-(((x0 - c_loc.x)^2) / fwhm + ((y0 - c_loc.y)^2) / fwhm)) + T_cold
  end
end
# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------

Δt = 1e-4
t = 0.0
maxt = 1.0
iter = 0
maxiter = 33
io_interval = 0.01
io_next = io_interval
pvd = paraview_collection("full_sim")
save_vtk(solver, ρ, T, mesh, iter, t, casename, pvd)
CurvilinearDiffusion.update_conductivity!(solver, T, ρ, κ, cₚ)

# fill!(solver.aⁿ⁺¹, 10.0)
while true
  CurvilinearDiffusion.update_conductivity!(solver, T, ρ, κ, cₚ)
  # dt = CurvilinearDiffusion.max_dt(solver, mesh)
  # global Δt = CFL * dt

  L₂, ncycles, is_converged = CurvilinearDiffusion.solve!(solver, mesh, T, Δt)
  @printf "cycle: %i t: %.4e, L2: %.1e, subcycles: %i Δt: %.3e\n" iter t L₂ ncycles Δt

  # L₂, Linf = CurvilinearDiffusion.solve!(solver, mesh, T, Δt)
  # @printf "cycle: %i t: %.4e, L2: %.1e, L∞: %.1e Δt: %.3e\n" iter t L₂ Linf Δt

  if t + Δt > io_next
    save_vtk(solver, ρ, T, mesh, iter, t, casename, pvd)
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
vtk_save(pvd);
