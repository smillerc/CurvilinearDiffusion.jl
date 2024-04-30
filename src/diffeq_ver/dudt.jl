using DifferentialEquations, LinearAlgebra, SparseArrays
using KernelAbstractions, UnPack, CurvilinearGrids
using Plots
using Symbolics
using WriteVTK
using Printf
using BenchmarkTools
using TimerOutputs
using Polyester
using LinearAlgebra
using LinearAlgebra.BLAS
using MKL
using Glob
using IncompleteLU

BLAS.get_config()

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

  @info "t: $t"
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
  ni, nj = (501, 501)
  nhalo = 6
  x, y = wavy_grid(ni, nj)
  # x, y = uniform_grid(ni, nj)
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
  # println("diffusivity extrema: $(extrema(diffusivity))")
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
      f_ξη * (
        aᵢ₊₁ⱼ * (uᵢ₊₁ⱼ₊₁ - uᵢ₊₁ⱼ₋₁) - # ∂u/∂η
        aᵢ₋₁ⱼ * (uᵢ₋₁ⱼ₊₁ - uᵢ₋₁ⱼ₋₁)   # ∂u/∂η
      ) + # ∂/∂ξ
      f_ξη * (
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
  κ0 = 10.0
  return κ0 #* T^3
  #   end
end

# thermal_conduction_2d = ODEProblem(du!, T, (0.0, 0.1), params);

function get_sparsity(mesh)
  ni, nj = cellsize_withhalo(mesh)
  len = ni * nj

  A = spdiagm(
    -ni - 1 => zeros(len - ni - 1), # (i-1, j-1)
    -ni => zeros(len - ni),     # (i  , j-1)
    -ni + 1 => zeros(len - ni + 1), # (i+1, j-1)
    -1 => zeros(len - 1),      # (i-1, j  )
    0 => ones(len),           # (i  , j  )
    1 => zeros(len - 1),      # (i+1, j  )
    ni - 1 => zeros(len - ni + 1), # (i-1, j+1)
    ni => zeros(len - ni),     # (i  , j+1)
    ni + 1 => zeros(len - ni - 1), # (i+1, j+1)
  )
  return A
end

function init_prob()
  @info "Initializing"
  mesh = initialize_mesh()
  # Temperature and density
  T_hot = 1e2
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

      u0[i, j] =
        T_hot * exp(-(((x0 - c_loc.x)^2) / fwhm + ((y0 - c_loc.y)^2) / fwhm)) + T_cold
    end
  end

  #   update_conductivity!(diffusivity, u0, ρ, κ, cₚ)
  params = (; mesh, density=ρ, diffusivity, source_term, meanfunc=harmonic_mean, κ, cₚ)

  # set up sparsity
  # du0 = copy(u0)
  # jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> du!(du, u, params, 0.0), du0, u0)
  jac_sparsity = get_sparsity(mesh)
  f = ODEFunction{true}(du!; jac_prototype=jac_sparsity)
  prob = ODEProblem{true}(f, u0, (0.0, 1.0), params)
  # # prob = ODEProblem{true}(du!, u0, (0.0, 1.0), params)

  # #   alg = Trapezoid(; linsolve=KrylovJL_GMRES(), precs=incompletelu, concrete_jac=true)

  # # ImplicitMidpoint, TRBDF2

  alg = Trapezoid(;
    # linsolve=UMFPACKFactorization(),
    # linsolve=MKLLUFactorization(),
    linsolve=KrylovJL_GMRES(; history=true),
    # concrete_jac=true,
    # precs=algebraicmultigrid,
  )
  dt = 1e-4
  integrator = init(prob, alg; dt=dt, save_everystep=false)

  @info "Done initializing"
  return integrator
end

function runme(prob, maxiter=Inf)
  global Δt = 1e-5
  global t = 0.0
  global maxt = 0.05
  global iter = 0
  global io_interval = 5e-4
  global io_next = io_interval
  pvd = paraview_collection("full_sim")
  #   @timeit "save_vtk" save_vtk(solver, ρ, T, mesh, iter, t, casename, pvd)
  save_vtk(prob.u, prob.p.mesh, iter, prob.t, "diffeq_ver", pvd)

  while true
    if iter == 1
      @info "reseting"
      reset_timer!()
    end
    @printf "cycle: %i t: %.4e, Δt: %.3e\n" iter prob.t prob.dt

    # fill!(prob.p.diffusivity, iter)
    # @info "conductivity"
    # @timeit "update_conductivity!"
    update_conductivity!(prob.p.diffusivity, prob.u, prob.p.density, κ, prob.p.cₚ)
    # @info "step"
    # @timeit "step!" 
    step!(prob, Δt, true)
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
  return prob
end

cd(@__DIR__)
rm.(glob("*.vts"))

prob = init_prob();
@profview begin
  p = runme(prob, 15)
end
