module ADESolverType

using CurvilinearGrids
using Polyester, StaticArrays
using UnPack

export ADESolver, solve!
export update_conductivity!, update_mesh_metrics!

include("averaging.jl")
include("boundary_conditions.jl")

struct ADESolver{T,N,EM,F,NT,BC}
  qⁿ⁺¹::Array{T,N}
  pⁿ⁺¹::Array{T,N}
  J::Array{T,N} # cell-centered Jacobian
  edge_metrics::Array{EM,N}
  diffusivity::Array{T,N} # cell-centered diffusivity
  source_term::Array{T,N} # cell-centered source term
  mean_func::F
  limits::NT
  bcs::BC
  nhalo::Int
end

"""
The Alternating Direction Explicit (ADE) diffusion solver. Provide the mesh
to give sizes to pre-allocate the type members. Provide a `mean_func` to
tell the solver how to determine diffusivity at the cell edges, i.e. via a
harmonic mean or arithmetic mean.
"""
function ADESolver(mesh::CurvilinearGrid2D, bcs, mean_func=arithmetic_mean, T=Float64)
  celldims = cellsize_withhalo(mesh)
  qⁿ⁺¹ = zeros(T, celldims)
  pⁿ⁺¹ = zeros(T, celldims)
  J = zeros(T, celldims)

  metric_type = typeof(_metrics_2d(mesh, 1, 1))
  edge_metrics = Array{metric_type,2}(undef, celldims)

  diffusivity = zeros(T, celldims)
  source_term = zeros(T, celldims)
  limits = mesh.limits

  solver = ADESolver(
    qⁿ⁺¹,
    pⁿ⁺¹,
    J,
    edge_metrics,
    diffusivity,
    source_term,
    mean_func,
    limits,
    bcs,
    mesh.nhalo,
  )
  update_mesh_metrics!(solver, mesh)

  return solver
end

"""Update the mesh metrics. Only do this whenever the mesh moves"""
function update_mesh_metrics!(solver, mesh::CurvilinearGrid2D)
  @unpack ilo, ihi, jlo, jhi = mesh.limits

  @inline for j in jlo:jhi
    for i in ilo:ihi
      solver.J[i, j] = jacobian(mesh, (i, j))
      solver.edge_metrics[i, j] = _metrics_2d(mesh, i, j)
    end
  end

  return nothing
end

@inline function _metrics_2d(mesh, i, j)
  metrics_i_plus_half = metrics_with_jacobian(mesh, (i + 1, j))
  metrics_i_minus_half = metrics_with_jacobian(mesh, (i, j))
  metrics_j_plus_half = metrics_with_jacobian(mesh, (i, j + 1))
  metrics_j_minus_half = metrics_with_jacobian(mesh, (i, j))
  # metrics_i_plus_half = metrics_with_jacobian(mesh, (i + 1 / 2, j))
  # metrics_i_minus_half = metrics_with_jacobian(mesh, (i - 1 / 2, j))
  # metrics_j_plus_half = metrics_with_jacobian(mesh, (i, j + 1 / 2))
  # metrics_j_minus_half = metrics_with_jacobian(mesh, (i, j - 1 / 2))

  return (
    Jᵢ₊½=metrics_i_plus_half.J,
    Jξx_ᵢ₊½=metrics_i_plus_half.ξx * metrics_i_plus_half.J,
    Jξy_ᵢ₊½=metrics_i_plus_half.ξy * metrics_i_plus_half.J,
    Jηx_ᵢ₊½=metrics_i_plus_half.ηx * metrics_i_plus_half.J,
    Jηy_ᵢ₊½=metrics_i_plus_half.ηy * metrics_i_plus_half.J,
    Jᵢ₋½=metrics_i_minus_half.J,
    Jξx_ᵢ₋½=metrics_i_minus_half.ξx * metrics_i_minus_half.J,
    Jξy_ᵢ₋½=metrics_i_minus_half.ξy * metrics_i_minus_half.J,
    Jηx_ᵢ₋½=metrics_i_minus_half.ηx * metrics_i_minus_half.J,
    Jηy_ᵢ₋½=metrics_i_minus_half.ηy * metrics_i_minus_half.J,
    Jⱼ₊½=metrics_j_plus_half.J,
    Jξx_ⱼ₊½=metrics_j_plus_half.ξx * metrics_j_plus_half.J,
    Jξy_ⱼ₊½=metrics_j_plus_half.ξy * metrics_j_plus_half.J,
    Jηx_ⱼ₊½=metrics_j_plus_half.ηx * metrics_j_plus_half.J,
    Jηy_ⱼ₊½=metrics_j_plus_half.ηy * metrics_j_plus_half.J,
    Jⱼ₋½=metrics_j_minus_half.J,
    Jξx_ⱼ₋½=metrics_j_minus_half.ξx * metrics_j_minus_half.J,
    Jξy_ⱼ₋½=metrics_j_minus_half.ξy * metrics_j_minus_half.J,
    Jηx_ⱼ₋½=metrics_j_minus_half.ηx * metrics_j_minus_half.J,
    Jηy_ⱼ₋½=metrics_j_minus_half.ηy * metrics_j_minus_half.J,
  )
end

"""
When using the diffusion solver to solve the heat equation, this can be used to
update the thermal conductivity of the mesh at each cell-center.

# Arguments
 - `solver::ADESolver`: The solver type
 - `T::Array`: Temperature
 - `ρ::Array`: Density
 - `κ::Function`: Function to determine thermal conductivity, e.g. κ(ρ,T) = κ0 * ρ * T^(5/2)
 - `cₚ::Real`: Heat capacity
"""
function update_conductivity!(solver::ADESolver, T::Array, ρ::Array, κ::Function, cₚ::Real)
  @inline for idx in eachindex(solver.diffusivity)
    rho = ρ[idx]
    kappa = κ(rho, T[idx])
    solver.diffusivity[idx] = kappa / (rho * cₚ)
  end

  return nothing
end

# function update_diffusivity(ADESolver::solver, κ)
#   @inline for idx in eachindex(solver.diffusivity)
#   end
# end

"""
# Arguments
 - α: Diffusion coefficient
"""
function solve!(solver::ADESolver, u, Δt)
  @unpack ilo, ihi, jlo, jhi = solver.limits

  α = solver.diffusivity

  # make alias for code readibilty
  # uⁿ⁺¹ = solver.uⁿ⁺¹  # new value of u
  pⁿ⁺¹ = solver.pⁿ⁺¹
  qⁿ⁺¹ = solver.qⁿ⁺¹

  applybc!(u, solver.bcs, solver.nhalo) # update the ghost cell temperatures

  @inline for idx in eachindex(u)
    pⁿ⁺¹[idx] = u[idx]
    qⁿ⁺¹[idx] = u[idx]
  end
  pⁿ = @views pⁿ⁺¹
  qⁿ = @views qⁿ⁺¹

  # Forward sweep ("implicit" pⁿ⁺¹ for i-1, j-1)
  for j in jlo:jhi
    for i in ilo:ihi
      Jᵢⱼ = solver.J[i, j]
      Js = Jᵢⱼ * solver.source_term[i, j]

      a_edge = (
        αᵢ₊½=solver.mean_func(α[i, j], α[i + 1, j]),
        αᵢ₋½=solver.mean_func(α[i, j], α[i - 1, j]),
        αⱼ₊½=solver.mean_func(α[i, j], α[i, j + 1]),
        αⱼ₋½=solver.mean_func(α[i, j], α[i, j - 1]),
      )

      begin
        @unpack fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½ = edge_terms_with_nonorthongal(
          a_edge, solver.edge_metrics[i, j]
        )
      end

      # Gᵢ₊½ = gᵢ₊½ * (pⁿ[i, j + 1] - pⁿ[i, j - 1] + pⁿ[i + 1, j + 1] - pⁿ[i + 1, j - 1])
        # Gᵢ₋½ = gᵢ₋½ * (pⁿ[i, j + 1] - pⁿ[i, j - 1] + pⁿ[i - 1, j + 1] - pⁿ[i - 1, j - 1])
        # Gⱼ₊½ = gⱼ₊½ * (pⁿ[i + 1, j] - pⁿ[i - 1, j] + pⁿ[i + 1, j + 1] - pⁿ[i - 1, j + 1])
        # Gⱼ₋½ = gⱼ₋½ * (pⁿ[i + 1, j] - pⁿ[i - 1, j] + pⁿ[i + 1, j - 1] - pⁿ[i - 1, j - 1])
        # Gᵢ₊½ = gᵢ₊½ * (pⁿ[i, j+1] - pⁿ⁺¹[i, j-1] + pⁿ[i+1, j+1] - pⁿ⁺¹[i+1, j-1])
        # Gᵢ₋½ = gᵢ₋½ * (pⁿ[i, j+1] - pⁿ⁺¹[i, j-1] + pⁿ⁺¹[i-1, j+1] - pⁿ⁺¹[i-1, j-1])
        # Gⱼ₊½ = gⱼ₊½ * (pⁿ[i+1, j] - pⁿ⁺¹[i-1, j] + pⁿ[i+1, j+1] - pⁿ⁺¹[i-1, j+1])
        # Gⱼ₋½ = gⱼ₋½ * (pⁿ[i+1, j] - pⁿ⁺¹[i-1, j] + pⁿ⁺¹[i+1, j-1] - pⁿ⁺¹[i-1, j-1])

      pⁿ⁺¹[i, j] = (
        (
          pⁿ[i, j] +
          (Δt / Jᵢⱼ) * (
            fᵢ₊½ * (pⁿ[i + 1, j] - pⁿ[i, j]) + #
            fⱼ₊½ * (pⁿ[i, j + 1] - pⁿ[i, j])   # current n level
            +
            fᵢ₋½ * pⁿ⁺¹[i - 1, j] +
            fⱼ₋½ * pⁿ⁺¹[i, j - 1] # n+1 level
            # +            Gᵢ₊½ - Gᵢ₋½ + Gⱼ₊½ - Gⱼ₋½ # non-orthongonal terms at n
            +
            Js
          )
        ) / (1 + (Δt / Jᵢⱼ) * (fᵢ₋½ + fⱼ₋½))
      )
    end
  end

  # Reverse sweep ("implicit" pⁿ⁺¹ for i+1, j+1)
  for j in jhi:-1:jlo
    for i in ihi:-1:ilo
      Jᵢⱼ = solver.J[i, j]
      Js = Jᵢⱼ * solver.source_term[i, j]

      a_edge = (
        αᵢ₊½=solver.mean_func(α[i, j], α[i + 1, j]),
        αᵢ₋½=solver.mean_func(α[i, j], α[i - 1, j]),
        αⱼ₊½=solver.mean_func(α[i, j], α[i, j + 1]),
        αⱼ₋½=solver.mean_func(α[i, j], α[i, j - 1]),
      )

      begin
        @unpack fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½ = edge_terms_with_nonorthongal(
          a_edge, solver.edge_metrics[i, j]
        )
      end

      # Gᵢ₊½ = gᵢ₊½ * (qⁿ⁺¹[i, j + 1] - qⁿ⁺¹[i, j - 1] + qⁿ⁺¹[i + 1, j + 1] - qⁿ⁺¹[i + 1, j - 1])
      # Gᵢ₋½ = gᵢ₋½ * (qⁿ⁺¹[i, j + 1] - qⁿ⁺¹[i, j - 1] + qⁿ⁺¹[i - 1, j + 1] - qⁿ⁺¹[i - 1, j - 1])
      # Gⱼ₊½ = gⱼ₊½ * (qⁿ⁺¹[i + 1, j] - qⁿ⁺¹[i - 1, j] + qⁿ⁺¹[i + 1, j + 1] - qⁿ⁺¹[i - 1, j + 1])
      # Gⱼ₋½ = gⱼ₋½ * (qⁿ⁺¹[i + 1, j] - qⁿ⁺¹[i - 1, j] + qⁿ⁺¹[i + 1, j - 1] - qⁿ⁺¹[i - 1, j - 1])
      # Gᵢ₊½ = gᵢ₊½ * (qⁿ⁺¹[i, j+1] - qⁿ[i, j-1] + qⁿ⁺¹[i+1, j+1] - qⁿ⁺¹[i+1, j-1])
      # Gᵢ₋½ = gᵢ₋½ * (qⁿ⁺¹[i, j+1] - qⁿ[i, j-1] + qⁿ⁺¹[i-1, j+1] - qⁿ[i-1, j-1])
      # Gⱼ₊½ = gⱼ₊½ * (qⁿ⁺¹[i+1, j] - qⁿ[i-1, j] + qⁿ⁺¹[i+1, j+1] - qⁿ⁺¹[i-1, j+1])
      # Gⱼ₋½ = gⱼ₋½ * (qⁿ⁺¹[i+1, j] - qⁿ[i-1, j] + qⁿ⁺¹[i+1, j-1] - qⁿ[i-1, j-1])

      qⁿ⁺¹[i, j] = (
        (
          qⁿ[i, j] +
          (Δt / Jᵢⱼ) * (
            -fᵢ₋½ * (qⁿ[i, j] - qⁿ[i - 1, j]) - #
            fⱼ₋½ * (qⁿ[i, j] - qⁿ[i, j - 1])    # current n level
            +
            fᵢ₊½ * qⁿ⁺¹[i + 1, j] +
            fⱼ₊½ * qⁿ⁺¹[i, j + 1] # n+1 level
            # +            Gᵢ₊½ - Gᵢ₋½ + Gⱼ₊½ - Gⱼ₋½ # non-orthongonal terms at n
            +
            Js
          )
        ) / (1 + (Δt / Jᵢⱼ) * (fᵢ₊½ + fⱼ₊½))
      )
    end
  end

  # Now average the forward/reverse sweeps
  L₂ = 0.0
  Linf = -Inf
  for j in jlo:jhi
    for i in ilo:ihi
      ϵ = abs(qⁿ⁺¹[i, j] - pⁿ⁺¹[i, j])
      Linf = max(Linf, ϵ)

      L₂ += ϵ * ϵ
      u[i, j] = 0.5(qⁿ⁺¹[i, j] + pⁿ⁺¹[i, j])
    end
  end

  N = (ihi - ilo + 1) * (jhi - jlo + 1)
  L₂ = sqrt(L₂ / N)

  return L₂, Linf
end

@inline function edge_terms_with_nonorthongal(α_edge, edge_metrics)
  @unpack αᵢ₊½, αᵢ₋½, αⱼ₊½, αⱼ₋½ = α_edge

  edge_metric = @views edge_metrics
  Jᵢ₊½ = edge_metric.Jᵢ₊½
  Jᵢ₋½ = edge_metric.Jᵢ₋½
  Jⱼ₊½ = edge_metric.Jⱼ₊½
  Jⱼ₋½ = edge_metric.Jⱼ₋½
  Jξx_ᵢ₊½ = edge_metric.Jξx_ᵢ₊½
  Jξy_ᵢ₊½ = edge_metric.Jξy_ᵢ₊½
  Jηx_ᵢ₊½ = edge_metric.Jηx_ᵢ₊½
  Jηy_ᵢ₊½ = edge_metric.Jηy_ᵢ₊½
  Jξx_ᵢ₋½ = edge_metric.Jξx_ᵢ₋½
  Jξy_ᵢ₋½ = edge_metric.Jξy_ᵢ₋½
  Jηx_ᵢ₋½ = edge_metric.Jηx_ᵢ₋½
  Jηy_ᵢ₋½ = edge_metric.Jηy_ᵢ₋½
  Jξx_ⱼ₊½ = edge_metric.Jξx_ⱼ₊½
  Jξy_ⱼ₊½ = edge_metric.Jξy_ⱼ₊½
  Jηx_ⱼ₊½ = edge_metric.Jηx_ⱼ₊½
  Jηy_ⱼ₊½ = edge_metric.Jηy_ⱼ₊½
  Jξx_ⱼ₋½ = edge_metric.Jξx_ⱼ₋½
  Jξy_ⱼ₋½ = edge_metric.Jξy_ⱼ₋½
  Jηx_ⱼ₋½ = edge_metric.Jηx_ⱼ₋½
  Jηy_ⱼ₋½ = edge_metric.Jηy_ⱼ₋½

  fᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
  fᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2) / Jᵢ₋½
  fⱼ₊½ = αⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½
  fⱼ₋½ = αⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2) / Jⱼ₋½

  gᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½ * Jηx_ᵢ₊½ + Jξy_ᵢ₊½ * Jηy_ᵢ₊½) / (4Jᵢ₊½)
  gᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½ * Jηx_ᵢ₋½ + Jξy_ᵢ₋½ * Jηy_ᵢ₋½) / (4Jᵢ₋½)
  gⱼ₊½ = αⱼ₊½ * (Jξx_ⱼ₊½ * Jηx_ⱼ₊½ + Jξy_ⱼ₊½ * Jηy_ⱼ₊½) / (4Jⱼ₊½)
  gⱼ₋½ = αⱼ₋½ * (Jξx_ⱼ₋½ * Jηx_ⱼ₋½ + Jξy_ⱼ₋½ * Jηy_ⱼ₋½) / (4Jⱼ₋½)

  return (; fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½)
end

end
