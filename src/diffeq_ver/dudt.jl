using DifferentialEquations, LinearAlgebra, SparseArrays
using KernelAbstractions, UnPack, CurvilinearGrids
using Plots
using Symbolics
using WriteVTK
using Printf
using BenchmarkTools
using TimerOutputs
using Polyester
using IncompleteLU

include("non_cons_terms.jl")

get_filename(iteration) = file_prefix * @sprintf("%07i", iteration)
function get_filename(iteration, name::String)
  name = replace(name, " " => "_")
  if !endswith(name, "_")
    name = name * "_"
  end
  return name * @sprintf("%07i", iteration)
end

function save_vtk(T, mesh, iteration, t, name, pvd)
  fn = get_filename(iteration, name)
  @info "Writing to $fn"
  ilo, ihi, jlo, jhi = mesh.domain_limits.cell

  #   α = @views solver.α[ilo:ihi, jlo:jhi]
  #   dens = @views ρ[ilo:ihi, jlo:jhi]
  temp = @views T[ilo:ihi, jlo:jhi]
  #   block = zeros(Int, size(dens))
  #   q = @views solver.source_term[ilo:ihi, jlo:jhi]
  #   kappa = α .* dens

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
    # vtk["density"] = dens
    vtk["temperature"] = temp
    # vtk["diffusivity"] = α
    # vtk["conductivity"] = kappa
    # vtk["block"] = block
    # vtk["source_term"] = q

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

function incompletelu(W, du, u, p, t, newW, Plprev, Prprev, solverdata)
  if newW === nothing || newW
    Pl = ilu(convert(AbstractMatrix, W); τ=0.5)
  else
    Pl = Plprev
  end
  return Pl, nothing
end

# Required due to a bug in Krylov.jl: https://github.com/JuliaSmoothOptimizers/Krylov.jl/pull/477
Base.eltype(::IncompleteLU.ILUFactorization{Tv,Ti}) where {Tv,Ti} = Tv

using AlgebraicMultigrid
function algebraicmultigrid(W, du, u, p, t, newW, Plprev, Prprev, solverdata)
  if newW === nothing || newW
    Pl = aspreconditioner(ruge_stuben(convert(AbstractMatrix, W)))
  else
    Pl = Plprev
  end
  return Pl, nothing
end

# ------------------------------------------------------------
# Grid Construction
# ------------------------------------------------------------

function wavy_grid(ni, nj)
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
  ni, nj = (1501, 1501)
  nhalo = 6
  x, y = wavy_grid(ni, nj)
  #   x, y = uniform_grid(ni, nj)
  return CurvilinearGrid2D(x, y, (ni, nj), nhalo)
end

function update_conductivity!(diffusivity, temp, ρ, κ, cₚ::Real)
  backend = KernelAbstractions.get_backend(diffusivity)
  conductivity_kernel(backend)(diffusivity, temp, ρ, κ, cₚ; ndrange=size(temp))

  return nothing
end

# super-simple kernel to update the diffusivity if
# we know the conductivity function
@kernel function conductivity_kernel(α, temperature, density, κ::F, cₚ) where {F}
  idx = @index(Global)

  @inbounds begin
    rho = density[idx]
    α[idx] = κ(rho, temperature[idx]) / (rho * cₚ)
  end
end

@inline harmonic_mean(a, b) = (2a * b) / (a + b)

function du!(dudt, u::AbstractArray{T,N}, params, t) where {T,N}

  #
  @unpack mesh, density, diffusivity, source_term, meanfunc, κ, cₚ = params

  #   update_conductivity!(diffusivity, u, density, κ, cₚ)

  @batch for idx in mesh.iterators.cell.domain
    @unpack α, β, f_ξ², f_η², f_ξη = non_cons_terms(
      mesh.cell_center_metrics, mesh.edge_metrics, idx
    )
    i, j = Tuple(idx)

    uᵢⱼ = u[i, j]
    uᵢ₊₁ⱼ = u[i + 1, j]
    uᵢ₋₁ⱼ = u[i - 1, j]
    uᵢⱼ₊₁ = u[i, j + 1]
    uᵢⱼ₋₁ = u[i, j - 1]

    uᵢ₊₁ⱼ₋₁ = u[i + 1, j - 1]
    uᵢ₋₁ⱼ₊₁ = u[i - 1, j + 1]
    uᵢ₊₁ⱼ₊₁ = u[i + 1, j + 1]
    uᵢ₋₁ⱼ₋₁ = u[i - 1, j - 1]

    # k0 = 1e12
    # aᵢⱼ = k0 * uᵢⱼ^3
    # aᵢ₊₁ⱼ = k0 * uᵢ₊₁ⱼ^3
    # aᵢ₋₁ⱼ = k0 * uᵢ₋₁ⱼ^3
    # aᵢⱼ₊₁ = k0 * uᵢⱼ₊₁^3
    # aᵢⱼ₋₁ = k0 * uᵢⱼ₋₁^3

    aᵢⱼ = diffusivity[i, j]
    aᵢ₊₁ⱼ = diffusivity[i + 1, j]
    aᵢ₋₁ⱼ = diffusivity[i - 1, j]
    aᵢⱼ₊₁ = diffusivity[i, j + 1]
    aᵢⱼ₋₁ = diffusivity[i, j - 1]

    aᵢ₊½ = meanfunc(aᵢⱼ, aᵢ₊₁ⱼ)
    aᵢ₋½ = meanfunc(aᵢⱼ, aᵢ₋₁ⱼ)
    aⱼ₊½ = meanfunc(aᵢⱼ, aᵢⱼ₊₁)
    aⱼ₋½ = meanfunc(aᵢⱼ, aᵢⱼ₋₁)

    dudt[i, j] =
      f_ξ² * (aᵢ₊½ * (uᵢ₊₁ⱼ - uᵢⱼ) - aᵢ₋½ * (uᵢⱼ - uᵢ₋₁ⱼ)) +
      f_η² * (aⱼ₊½ * (uᵢⱼ₊₁ - uᵢⱼ) - aⱼ₋½ * (uᵢⱼ - uᵢⱼ₋₁)) +
      +f_ξη * (
        aᵢ₊₁ⱼ * (uᵢ₊₁ⱼ₊₁ - uᵢ₊₁ⱼ₋₁) - # ∂u/∂η
        aᵢ₋₁ⱼ * (uᵢ₋₁ⱼ₊₁ - uᵢ₋₁ⱼ₋₁)   # ∂u/∂η
      ) + # ∂/∂ξ
      +f_ξη * (
        aᵢⱼ₊₁ * (uᵢ₊₁ⱼ₊₁ - uᵢ₋₁ⱼ₊₁) - # ∂u/∂ξ
        aᵢⱼ₋₁ * (uᵢ₊₁ⱼ₋₁ - uᵢ₋₁ⱼ₋₁)   # ∂u/∂ξ
      ) + # ∂/∂η
      aᵢⱼ * α / 2 * (uᵢ₊₁ⱼ - uᵢ₋₁ⱼ) +
      aᵢⱼ * β / 2 * (uᵢⱼ₊₁ - uᵢⱼ₋₁) +
      source_term[i, j]
  end

  return nothing
end

# Define the conductivity model
@inline function κ(ρ, T)
  #   if !isfinite(T)
  #     return 0.0
  #   else
  κ0 = 1.0
  return κ0 * T^3
  #   end
end

# thermal_conduction_2d = ODEProblem(du!, T, (0.0, 0.1), params);

function init_prob()
  mesh = initialize_mesh()
  # Temperature and density
  T_hot = 1e3
  T_cold = 1e-2
  u0 = ones(Float64, cellsize_withhalo(mesh)) * T_cold
  ρ = ones(Float64, cellsize_withhalo(mesh))
  diffusivity = ones(Float64, cellsize_withhalo(mesh))
  cₚ = 1.0

  # Gaussian source term
  fwhm = 0.5
  x0 = 0.0
  y0 = 0.0
  @unpack ilo, ihi, jlo, jhi = mesh.domain_limits.cell

  source_term = zeros(Float64, cellsize_withhalo(mesh))

  for j in jlo:jhi
    for i in ilo:ihi
      c_loc = centroid(mesh, (i, j))

      source_term[i, j] =
        T_hot * exp(-(((x0 - c_loc.x)^2) / fwhm + ((y0 - c_loc.y)^2) / fwhm)) + T_cold
    end
  end

  #   update_conductivity!(diffusivity, u0, ρ, κ, cₚ)
  params = (; mesh, density=ρ, diffusivity, source_term, meanfunc=harmonic_mean, κ, cₚ)

  # set up sparsity
  du0 = copy(u0)
  jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> du!(du, u0, params, 0.0), du0, u0)

  f = ODEFunction{true}(du!; jac_prototype=jac_sparsity)
  prob = ODEProblem(f, u0, (0.0, 1.0), params)

  #   alg = Trapezoid(; linsolve=KrylovJL_GMRES(), precs=incompletelu, concrete_jac=true)
  alg = Trapezoid(; linsolve=UMFPACKFactorization())
  integrator = init(prob, alg; save_everystep=false)

  #   prob = ODEProblem(du!, u0, (0.0, 1.0), params)
  #   integrator = init(prob, Trapezoid(; linsolve=linsolver); save_everystep=false)

  #   integrator = init(
  #     prob,
  #     Trapezoid(; linsolve=KrylovJL_GMRES(), precs=algebraicmultigrid, concrete_jac=true);
  #     save_everystep=false,
  #   )

  # integrator = init(prob, Trapezoid(; linsolve=KLUFactorization()); save_everystep=false)
  #   integrator = init(
  #     prob, Trapezoid(; linsolve=UMFPACKFactorization()); save_everystep=false
  #   )
  #   integrator = init(prob, KenCarp47(; linsolve=KLUFactorization()); save_everystep=false)
  return integrator
  #   return nothing
end

# @code_warntype init_prob();
# alg = ImplicitEuler(; linsolve=KrylovJL_GMRES())
# integrator = init(prob, alg);
# @code_warntype init(prob, alg)
# prob = init_prob();

# dt = 1e-8
# step!(prob, dt, true)
# # begin
#   @info "Initializing"
#   prob = init_prob()
#   #   @info "update conductivity"
#   #   update_conductivity!(diffusivity, T, ρ, κ, cₚ)
#   #   @info "step!"
#   #   step!(prob, dt, true)
#   #   @info "benchmarking"
#   #   @btime step!($prob, $dt, $true)
# end

# update_conductivity!(prob.p.diffusivity, prob.u, prob.p.density, κ, prob.p.cₚ)
# prob.p.diffusivity |> extrema

# begin
#   global t = 0
#   global iter = 0
#   while true
#     println("iter: $iter")
#     iter += 1
#     dt = 1e-8
#     t += dt
#     # update_conductivity!(prob.p.diffusivity, prob.u, prob.p.density, κ, prob.p.cₚ)
#     step!(prob, dt, true)
#     pvd = paraview_collection("full_sim")
#     if iter
#       save_vtk(prob.u, prob.p.mesh, iter, prob.t, "diffeq_ver", pvd)
#       if iter >= 100
#         break
#       end
#     end
#   end

begin
  prob = init_prob()
  global Δt = 1e-4
  global t = 0.0
  global maxt = 0.2
  global iter = 0
  global maxiter = 100
  global io_interval = 1e-3
  global io_next = io_interval
  pvd = paraview_collection("full_sim")
  #   @timeit "save_vtk" save_vtk(solver, ρ, T, mesh, iter, t, casename, pvd)
  save_vtk(prob.u, prob.p.mesh, iter, prob.t, "diffeq_ver", pvd)

  reset_timer!()
  while true
    @printf "cycle: %i t: %.4e, Δt: %.3e\n" iter t Δt

    # fill!(prob.p.diffusivity, iter)
    # update_conductivity!(prob.p.diffusivity, prob.u, prob.p.density, κ, prob.p.cₚ)
    @timeit "step!" step!(prob, Δt, true)
    # prob.p.diffusivity .= iter #.^ 3

    if t + Δt > io_next
      save_vtk(prob.u, prob.p.mesh, iter, prob.t, "diffeq_ver", pvd)
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

  print_timer()
end

# @profview step!(integrator, dt, true)

# # for (i, ut) in enumerate(prob.sol.u)
# @show prob.stats
# # end
