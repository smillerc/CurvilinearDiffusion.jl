module ADESolverType

using CurvilinearGrids
using Polyester, StaticArrays
using UnPack

export ADESolver, solve!
export update_conductivity!, update_mesh_metrics!

include("averaging.jl")
include("boundary_conditions.jl")

# TODO: make a linear and non-linear version based on κ or a

struct ADESolver{T,N,EM,F,NT,BC}
  qⁿ⁺¹::Array{T,N}
  pⁿ⁺¹::Array{T,N}
  J::Array{T,N} # cell-centered Jacobian
  metrics::Array{EM,N}
  a₋ⁿ⁺¹::Array{T,N} # cell-centered diffusivity
  a₊ⁿ⁺¹::Array{T,N} # cell-centered diffusivity
  aⁿ⁺¹::Array{T,N} # cell-centered diffusivity
  source_term::Array{T,N} # cell-centered source term
  mean_func::F
  limits::NT
  bcs::BC
  nhalo::Int
  conservative::Bool # uses the conservative form
end

"""
The Alternating Direction Explicit (ADE) diffusion solver. Provide the mesh
to give sizes to pre-allocate the type members. Provide a `mean_func` to
tell the solver how to determine diffusivity at the cell edges, i.e. via a
harmonic mean or arithmetic mean.
"""
function ADESolver(
  mesh::CurvilinearGrid2D, bcs, form=:conservative, mean_func=arithmetic_mean, T=Float64
)
  celldims = cellsize_withhalo(mesh)
  qⁿ⁺¹ = zeros(T, celldims)
  pⁿ⁺¹ = zeros(T, celldims)
  J = zeros(T, celldims)

  if form === :conservative
    conservative = true
    metric_type = typeof(_conservative_metrics_2d(mesh, 1, 1))
  else
    conservative = false
    metric_type = typeof(_non_conservative_metrics_2d(mesh, 1, 1))
  end
  @show conservative

  edge_metrics = Array{metric_type,2}(undef, celldims)

  diffusivity = zeros(T, celldims)
  diffusivity_forward = zeros(T, celldims)
  diffusivity_reverse = zeros(T, celldims)
  source_term = zeros(T, celldims)
  limits = mesh.limits

  solver = ADESolver(
    qⁿ⁺¹,
    pⁿ⁺¹,
    J,
    edge_metrics,
    diffusivity_forward,
    diffusivity_reverse,
    diffusivity,
    source_term,
    mean_func,
    limits,
    bcs,
    mesh.nhalo,
    conservative,
  )
  update_mesh_metrics!(solver, mesh)

  return solver
end

"""Update the mesh metrics. Only do this whenever the mesh moves"""
function update_mesh_metrics!(solver, mesh::CurvilinearGrid2D)
  @unpack ilo, ihi, jlo, jhi = mesh.limits

  if solver.conservative
    @inline for j in jlo:jhi
      for i in ilo:ihi
        solver.J[i, j] = jacobian(mesh, (i, j))
        solver.metrics[i, j] = _conservative_metrics_2d(mesh, i, j)
      end
    end
  else
    @inline for j in jlo:jhi
      for i in ilo:ihi
        solver.J[i, j] = jacobian(mesh, (i, j))
        solver.metrics[i, j] = _non_conservative_metrics_2d(mesh, i, j)
      end
    end
  end

  return nothing
end

@inline function _non_conservative_metrics_2d(mesh, i, j)
  # note, metrics(mesh, (i,j)) uses node-based indexing, and here we're making
  # a tuple of metrics that uses cell-based indexing, thus the weird 1/2 offsets
  metricsᵢⱼ = cell_metrics(mesh, (i, j))
  metricsᵢ₊½ = cell_metrics(mesh, (i + 1 / 2, j))
  metricsⱼ₊½ = cell_metrics(mesh, (i, j + 1 / 2))
  metricsᵢ₋½ = cell_metrics(mesh, (i - 1 / 2, j))
  metricsⱼ₋½ = cell_metrics(mesh, (i, j - 1 / 2))
  # metricsᵢⱼ = metrics(mesh, (i + 1 / 2, j + 1 / 2))
  # metricsᵢ₋½ = metrics(mesh, (i, j + 1 / 2))
  # metricsⱼ₋½ = metrics(mesh, (i + 1 / 2, j))
  # metricsᵢ₊½ = metrics(mesh, (i + 1, j + 1 / 2))
  # metricsⱼ₊½ = metrics(mesh, (i + 1 / 2, j + 1))

  return (
    ξx=metricsᵢⱼ.ξx,
    ξy=metricsᵢⱼ.ξy,
    ηx=metricsᵢⱼ.ηx,
    ηy=metricsᵢⱼ.ηy,
    ξxᵢ₋½=metricsᵢ₋½.ξx,
    ξyᵢ₋½=metricsᵢ₋½.ξy,
    ηxᵢ₋½=metricsᵢ₋½.ηx,
    ηyᵢ₋½=metricsᵢ₋½.ηy,
    ξxⱼ₋½=metricsⱼ₋½.ξx,
    ξyⱼ₋½=metricsⱼ₋½.ξy,
    ηxⱼ₋½=metricsⱼ₋½.ηx,
    ηyⱼ₋½=metricsⱼ₋½.ηy,
    ξxᵢ₊½=metricsᵢ₊½.ξx,
    ξyᵢ₊½=metricsᵢ₊½.ξy,
    ηxᵢ₊½=metricsᵢ₊½.ηx,
    ηyᵢ₊½=metricsᵢ₊½.ηy,
    ξxⱼ₊½=metricsⱼ₊½.ξx,
    ξyⱼ₊½=metricsⱼ₊½.ξy,
    ηxⱼ₊½=metricsⱼ₊½.ηx,
    ηyⱼ₊½=metricsⱼ₊½.ηy,
  )
end

@inline cutoff(a) = (0.5(abs(a) + a))

@inline function _conservative_metrics_2d(mesh, i, j)
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
  @inline for idx in eachindex(T)
    rho = ρ[idx]
    kappa = κ(rho, T[idx])
    solver.aⁿ⁺¹[idx] = kappa / (rho * cₚ)
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
function solve!(solver::ADESolver, mesh, u, Δt)
  if solver.conservative
    solve_conservative!(solver, mesh, u, Δt)
  else
    solve_nc_nonlinear!(solver, mesh, u, Δt)
  end
end
# solve!(solver::ADESolver, mesh, u, Δt) = solve_nc_nonlinear!(solver, mesh, u, Δt)
# solve!(solver::ADESolver, mesh, u, Δt) = solve_nc_nonlinear_explicit!(solver, mesh, u, Δt)

function solve_conservative!(solver::ADESolver, mesh, u, Δt)
  @unpack ilo, ihi, jlo, jhi = solver.limits

  α = solver.aⁿ⁺¹

  # make alias for code readibilty
  # uⁿ⁺¹ = solver.uⁿ⁺¹  # new value of u
  pⁿ⁺¹ = solver.pⁿ⁺¹
  qⁿ⁺¹ = solver.qⁿ⁺¹

  applybc!(u, solver.bcs, solver.nhalo) # update the ghost cell temperatures

  @inline for idx in eachindex(u)
    pⁿ⁺¹[idx] = u[idx]
  end
  pⁿ = @views pⁿ⁺¹

  # Forward sweep ("implicit" pⁿ⁺¹ for i-1, j-1)
  for j in jlo:jhi
    for i in ilo:ihi
      Jᵢⱼ = solver.J[i, j]
      Js = Jᵢⱼ * solver.source_term[i, j]

      aᵢ₊½ = solver.mean_func(α[i, j], α[i + 1, j])
      aᵢ₋½ = solver.mean_func(α[i, j], α[i - 1, j])
      aⱼ₊½ = solver.mean_func(α[i, j], α[i, j + 1])
      aⱼ₋½ = solver.mean_func(α[i, j], α[i, j - 1])
      a_edge = (αᵢ₊½=aᵢ₊½, αᵢ₋½=aᵢ₋½, αⱼ₊½=aⱼ₊½, αⱼ₋½=aⱼ₋½)
      # a_edge = (
      #   αᵢ₊½=solver.mean_func((α[i, j], α[i + 1, j]), (solver.J[i, j], solver.J[i + 1, j])),
      #   αᵢ₋½=solver.mean_func((α[i, j], α[i - 1, j]), (solver.J[i, j], solver.J[i - 1, j])),
      #   αⱼ₊½=solver.mean_func((α[i, j], α[i, j + 1]), (solver.J[i, j], solver.J[i, j + 1])),
      #   αⱼ₋½=solver.mean_func((α[i, j], α[i, j - 1]), (solver.J[i, j], solver.J[i, j - 1])),
      # )

      begin
        @unpack fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½ = edge_terms_with_nonorthongal(
          a_edge, solver.metrics[i, j]
        )
      end

      Gᵢ₊½ = gᵢ₊½ * (pⁿ[i, j + 1] - pⁿ[i, j - 1] + pⁿ[i + 1, j + 1] - pⁿ[i + 1, j - 1])
      Gᵢ₋½ = gᵢ₋½ * (pⁿ[i, j + 1] - pⁿ[i, j - 1] + pⁿ[i - 1, j + 1] - pⁿ[i - 1, j - 1])
      Gⱼ₊½ = gⱼ₊½ * (pⁿ[i + 1, j] - pⁿ[i - 1, j] + pⁿ[i + 1, j + 1] - pⁿ[i - 1, j + 1])
      Gⱼ₋½ = gⱼ₋½ * (pⁿ[i + 1, j] - pⁿ[i - 1, j] + pⁿ[i + 1, j - 1] - pⁿ[i - 1, j - 1])
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
            +
            Gᵢ₊½ - Gᵢ₋½ + Gⱼ₊½ - Gⱼ₋½ # non-orthongonal terms at n
            + Js
          )
        ) / (1 + (Δt / Jᵢⱼ) * (fᵢ₋½ + fⱼ₋½))
      )
    end
  end

  @inline for idx in eachindex(u)
    qⁿ⁺¹[idx] = u[idx]
  end
  qⁿ = @views qⁿ⁺¹

  # Reverse sweep ("implicit" pⁿ⁺¹ for i+1, j+1)
  for j in jhi:-1:jlo
    for i in ihi:-1:ilo
      Jᵢⱼ = solver.J[i, j]
      Js = Jᵢⱼ * solver.source_term[i, j]

      aᵢ₊½ = solver.mean_func(α[i, j], α[i + 1, j])
      aᵢ₋½ = solver.mean_func(α[i, j], α[i - 1, j])
      aⱼ₊½ = solver.mean_func(α[i, j], α[i, j + 1])
      aⱼ₋½ = solver.mean_func(α[i, j], α[i, j - 1])
      a_edge = (αᵢ₊½=aᵢ₊½, αᵢ₋½=aᵢ₋½, αⱼ₊½=aⱼ₊½, αⱼ₋½=aⱼ₋½)
      # a_edge = (
      #   αᵢ₊½=solver.mean_func((α[i, j], α[i + 1, j]), (solver.J[i, j], solver.J[i + 1, j])),
      #   αᵢ₋½=solver.mean_func((α[i, j], α[i - 1, j]), (solver.J[i, j], solver.J[i - 1, j])),
      #   αⱼ₊½=solver.mean_func((α[i, j], α[i, j + 1]), (solver.J[i, j], solver.J[i, j + 1])),
      #   αⱼ₋½=solver.mean_func((α[i, j], α[i, j - 1]), (solver.J[i, j], solver.J[i, j - 1])),
      # )

      begin
        @unpack fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½ = edge_terms_with_nonorthongal(
          a_edge, solver.metrics[i, j]
        )
      end

      Gᵢ₊½ = gᵢ₊½ * (qⁿ[i, j + 1] - qⁿ[i, j - 1] + qⁿ[i + 1, j + 1] - qⁿ[i + 1, j - 1])
      Gᵢ₋½ = gᵢ₋½ * (qⁿ[i, j + 1] - qⁿ[i, j - 1] + qⁿ[i - 1, j + 1] - qⁿ[i - 1, j - 1])
      Gⱼ₊½ = gⱼ₊½ * (qⁿ[i + 1, j] - qⁿ[i - 1, j] + qⁿ[i + 1, j + 1] - qⁿ[i - 1, j + 1])
      Gⱼ₋½ = gⱼ₋½ * (qⁿ[i + 1, j] - qⁿ[i - 1, j] + qⁿ[i + 1, j - 1] - qⁿ[i - 1, j - 1])
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
            +
            Gᵢ₊½ - Gᵢ₋½ + Gⱼ₊½ - Gⱼ₋½ # non-orthongonal terms at n
            + Js
          )
        ) / (1 + (Δt / Jᵢⱼ) * (fᵢ₊½ + fⱼ₊½))
      )

      # if !isfinite(qⁿ⁺¹[i, j])
      #   @show (i, j)
      #   @show qⁿ⁺¹[i, j] qⁿ[i, j]
      #   @show a_edge
      #   @show u[i, j]
      #   @show u[i + 1, j]
      #   @show u[i, j + 1]
      #   @show u[i - 1, j]
      #   @show u[i, j - 1]
      #   @show α[i, j]
      #   @show α[i + 1, j]
      #   @show α[i, j + 1]
      #   @show α[i - 1, j]
      #   @show α[i, j - 1]
      #   @show fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½
      #   @show gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½
      #   error("qⁿ⁺¹ is not valid!")
      # end
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

  for j in jlo:jhi
    for i in ilo:ihi
      if !isfinite(u[i, j])
        @show u[i, j]
        @show qⁿ⁺¹[i, j]
        @show pⁿ⁺¹[i, j]
        error("Invalid value for u $(u[i,j]) at $((i,j))")
      end
    end
  end

  N = (ihi - ilo + 1) * (jhi - jlo + 1)
  L₂ = sqrt(L₂ / N)

  return L₂, Linf
end

function solve_nc_nonlinear!(solver::ADESolver, mesh, u, Δt)
  # The mesh metrics are indexed node-based, so the
  # convention can sometimes be confusing when trying to
  # get the cell-based values
  #
  # The node indexing is as follows:
  #
  #         (i,j+1)            (i+1,j+1)
  #           o---------X----------o
  #           |     (i+1/2,j+1)    |
  #           |                    |
  #           |      cell idx      |
  # (i,j+1/2) X       (I,J)        X (i+1,j+1/2)
  #           |                    |
  #           |                    |
  #           |     (i+1/2,j)      |
  #           o---------X----------o
  #         (i,j)               (1+1,j)

  @unpack ilo, ihi, jlo, jhi = solver.limits

  α = solver.aⁿ⁺¹
  # α⁻ = solver.a₋ⁿ⁺¹
  # α⁺ = solver.a₊ⁿ⁺¹
  # @inline for idx in eachindex(α)
  #   α⁻[idx] = α[idx]
  #   α⁺[idx] = α[idx]
  # end

  applybc!(u, solver.bcs, solver.nhalo) # update the ghost cell temperatures

  # @inline for idx in eachindex(α)
  #   κ0 = 1
  #   α[idx] = κ0 * u[idx]^3
  # end

  #------------------------------------------------------
  # Forward Sweep
  #------------------------------------------------------
  # anything at i-1 or j-1 is from time level n+1, since we already computed it

  # make alias for code readibilty
  pⁿ⁺¹ = solver.pⁿ⁺¹
  @inline for idx in eachindex(u)
    pⁿ⁺¹[idx] = u[idx]
  end
  pⁿ = @views pⁿ⁺¹

  for j in jlo:jhi
    for i in ilo:ihi
      aᵢ₊½ = solver.mean_func(α[i, j], α[i + 1, j])
      aᵢ₋½ = solver.mean_func(α[i, j], α[i - 1, j])
      aⱼ₊½ = solver.mean_func(α[i, j], α[i, j + 1])
      aⱼ₋½ = solver.mean_func(α[i, j], α[i, j - 1])

      aᵢⱼ = α[i, j]
      aᵢ₊₁ⱼ = α[i + 1, j]
      aᵢ₋₁ⱼ = α[i - 1, j]
      aᵢⱼ₊₁ = α[i, j + 1]
      aᵢⱼ₋₁ = α[i, j - 1]

      m = solver.metrics[i, j]
      αᵢⱼ = (
        m.ξx * (m.ξxᵢ₊½ - m.ξxᵢ₋½) +
        m.ξy * (m.ξyᵢ₊½ - m.ξyᵢ₋½) +
        m.ηx * (m.ξxⱼ₊½ - m.ξxⱼ₋½) +
        m.ηy * (m.ξyⱼ₊½ - m.ξyⱼ₋½)
      )

      βᵢⱼ = (
        m.ξx * (m.ηxᵢ₊½ - m.ηxᵢ₋½) +
        m.ξy * (m.ηyᵢ₊½ - m.ηyᵢ₋½) +
        m.ηx * (m.ηxⱼ₊½ - m.ηxⱼ₋½) +
        m.ηy * (m.ηyⱼ₊½ - m.ηyⱼ₋½)
      )

      pⁿ⁺¹[i, j] =
        (
          # orthogonal terms
          (m.ξx^2 + m.ξy^2) * (aᵢ₊½ * (pⁿ[i + 1, j] - pⁿ[i, j]) + aᵢ₋½ * pⁿ⁺¹[i - 1, j]) +
          (m.ηx^2 + m.ηy^2) * (aⱼ₊½ * (pⁿ[i, j + 1] - pⁿ[i, j]) + aⱼ₋½ * pⁿ⁺¹[i, j - 1]) +
          # non-orthogonal terms
          0.25(m.ξx * m.ηx + m.ξy * m.ηy) * (
            aᵢ₊₁ⱼ * (pⁿ[i + 1, j + 1] - pⁿ[i + 1, j - 1]) -
            aᵢ₋₁ⱼ * (pⁿ[i - 1, j + 1] - pⁿ[i - 1, j - 1])
          ) +
          0.25(m.ξx * m.ηx + m.ξy * m.ηy) * (
            aᵢⱼ₊₁ * (pⁿ[i + 1, j + 1] - pⁿ[i - 1, j + 1]) -
            aᵢⱼ₋₁ * (pⁿ[i + 1, j - 1] - pⁿ[i - 1, j - 1])
          ) +
          # geometric terms
          0.5aᵢⱼ * αᵢⱼ * (pⁿ[i + 1, j] - pⁿ⁺¹[i - 1, j]) +
          0.5aᵢⱼ * βᵢⱼ * (pⁿ[i, j + 1] - pⁿ⁺¹[i, j - 1]) +
          # source and remaining terms
          solver.source_term[i, j] +
          (pⁿ[i, j] / Δt)
        ) / ((1 / Δt) + (aᵢ₋½ * (m.ξx^2 + m.ξy^2) + aⱼ₋½ * (m.ηx^2 + m.ηy^2)))

      pⁿ⁺¹[i, j] = cutoff(pⁿ⁺¹[i, j])
      # update the diffusivity
      # α[i, j]  = kappa / (rho * cₚ)
      # κ0 = 1
      # α⁺[i, j] = κ0 * pⁿ⁺¹[i, j]^3
    end
  end

  #------------------------------------------------------
  # Reverse Sweep
  #------------------------------------------------------
  # anything at i+1 or j+1 is from time level n+1, since we already computed it

  # make alias for code readibilty
  qⁿ⁺¹ = solver.qⁿ⁺¹
  @inline for idx in eachindex(u)
    qⁿ⁺¹[idx] = u[idx]
  end
  qⁿ = @views qⁿ⁺¹

  for j in jhi:-1:jlo
    for i in ihi:-1:ilo
      aᵢ₊½ = solver.mean_func(α[i, j], α[i + 1, j])
      aᵢ₋½ = solver.mean_func(α[i, j], α[i - 1, j])
      aⱼ₊½ = solver.mean_func(α[i, j], α[i, j + 1])
      aⱼ₋½ = solver.mean_func(α[i, j], α[i, j - 1])

      aᵢⱼ = α[i, j]
      aᵢ₊₁ⱼ = α[i + 1, j]
      aᵢ₋₁ⱼ = α[i - 1, j]
      aᵢⱼ₊₁ = α[i, j + 1]
      aᵢⱼ₋₁ = α[i, j - 1]

      m = solver.metrics[i, j]

      αᵢⱼ = (
        m.ξx * (m.ξxᵢ₊½ - m.ξxᵢ₋½) +
        m.ξy * (m.ξyᵢ₊½ - m.ξyᵢ₋½) +
        m.ηx * (m.ξxⱼ₊½ - m.ξxⱼ₋½) +
        m.ηy * (m.ξyⱼ₊½ - m.ξyⱼ₋½)
      )

      βᵢⱼ = (
        m.ξx * (m.ηxᵢ₊½ - m.ηxᵢ₋½) +
        m.ξy * (m.ηyᵢ₊½ - m.ηyᵢ₋½) +
        m.ηx * (m.ηxⱼ₊½ - m.ηxⱼ₋½) +
        m.ηy * (m.ηyⱼ₊½ - m.ηyⱼ₋½)
      )

      qⁿ⁺¹[i, j] =
        (
          # orthogonal terms
          (m.ξx^2 + m.ξy^2) * (aᵢ₊½ * qⁿ⁺¹[i + 1, j] - aᵢ₋½ * (qⁿ[i, j] - qⁿ⁺¹[i - 1, j])) +
          (m.ηx^2 + m.ηy^2) * (aⱼ₊½ * qⁿ⁺¹[i, j + 1] - aⱼ₋½ * (qⁿ[i, j] - qⁿ⁺¹[i, j - 1])) +
          # non-orthogonal terms
          0.25(m.ξx * m.ηx + m.ξy * m.ηy) * (
            aᵢ₊₁ⱼ * (qⁿ[i + 1, j + 1] - qⁿ[i + 1, j - 1]) -
            aᵢ₋₁ⱼ * (qⁿ[i - 1, j + 1] - qⁿ[i - 1, j - 1])
          ) +
          0.25(m.ξx * m.ηx + m.ξy * m.ηy) * (
            aᵢⱼ₊₁ * (qⁿ[i + 1, j + 1] - qⁿ[i - 1, j + 1]) -
            aᵢⱼ₋₁ * (qⁿ[i + 1, j - 1] - qⁿ[i - 1, j - 1])
          ) +
          # geometric terms
          0.5aᵢⱼ * αᵢⱼ * (qⁿ[i + 1, j] - qⁿ⁺¹[i - 1, j]) +
          0.5aᵢⱼ * βᵢⱼ * (qⁿ[i, j + 1] - qⁿ⁺¹[i, j - 1]) +
          # source and remaining terms
          solver.source_term[i, j] +
          (qⁿ[i, j] / Δt)
        ) / ((1 / Δt) + (aᵢ₊½ * (m.ξx^2 + m.ξy^2) + aⱼ₊½ * (m.ηx^2 + m.ηy^2)))

      qⁿ⁺¹[i, j] = cutoff(qⁿ⁺¹[i, j])
      # update the diffusivity
      # α[i, j]  = kappa / (rho * cₚ)
      # κ0 = 1
      # α⁻[i, j] = κ0 * qⁿ⁺¹[i, j]^3
    end
  end

  # # Now average the forward/reverse sweeps
  # @inline for idx in eachindex(α)
  #   α[idx] = 0.5(α⁻[idx] + α⁺[idx])
  # end

  L₂ = 0.0
  Linf = -Inf
  for j in jlo:jhi
    for i in ilo:ihi
      ϵ = abs(qⁿ⁺¹[i, j] - pⁿ⁺¹[i, j])
      Linf = max(Linf, ϵ)

      L₂ += ϵ * ϵ
      u_ave = 0.5(qⁿ⁺¹[i, j] + pⁿ⁺¹[i, j])
      u[i, j] = u_ave
    end
  end

  for j in jlo:jhi
    for i in ilo:ihi
      if !isfinite(u[i, j]) || u[i, j] < 0
        @show u[i, j]
        @show qⁿ⁺¹[i, j]
        @show pⁿ⁺¹[i, j]
        error("Invalid value for u $(u[i,j]) at $((i,j))")
      end
    end
  end

  N = (ihi - ilo + 1) * (jhi - jlo + 1)
  L₂ = sqrt(L₂ / N)

  return L₂, Linf
end

function solve_nc_nonlinear_explicit!(solver::ADESolver, mesh, u, Δt)

  # The mesh metrics are indexed node-based, so the
  # convention can sometimes be confusing when trying to
  # get the cell-based values
  #
  # The node indexing is as follows:
  #
  #         (i,j+1)            (i+1,j+1)
  #           o---------X----------o
  #           |     (i+1/2,j+1)    |
  #           |                    |
  #           |      cell idx      |
  # (i,j+1/2) X       (I,J)        X (i+1,j+1/2)
  #           |                    |
  #           |                    |
  #           |     (i+1/2,j)      |
  #           o---------X----------o
  #         (i,j)               (1+1,j)

  @unpack ilo, ihi, jlo, jhi = solver.limits

  α = solver.aⁿ⁺¹
  # α⁻ = solver.a₋ⁿ⁺¹
  # α⁺ = solver.a₊ⁿ⁺¹
  # @inline for idx in eachindex(α)
  #   α⁻[idx] = α[idx]
  #   α⁺[idx] = α[idx]
  # end

  applybc!(u, solver.bcs, solver.nhalo) # update the ghost cell temperatures

  # @inline for idx in eachindex(α)
  #   κ0 = 1
  #   α[idx] = κ0 * u[idx]^3
  # end

  #------------------------------------------------------
  # Forward Sweep
  #------------------------------------------------------
  # anything at i+1 or j+1 is from time level n+1, since we already computed it

  # make alias for code readibilty
  uⁿ⁺¹ = solver.pⁿ⁺¹

  for j in jlo:jhi
    for i in ilo:ihi
      aᵢ₊½ = solver.mean_func(α[i, j], α[i + 1, j])
      aᵢ₋½ = solver.mean_func(α[i, j], α[i - 1, j])
      aⱼ₊½ = solver.mean_func(α[i, j], α[i, j + 1])
      aⱼ₋½ = solver.mean_func(α[i, j], α[i, j - 1])

      aᵢⱼ = α[i, j]
      aᵢ₊₁ⱼ = α[i + 1, j]
      aᵢ₋₁ⱼ = α[i - 1, j]
      aᵢⱼ₊₁ = α[i, j + 1]
      aᵢⱼ₋₁ = α[i, j - 1]

      m = solver.metrics[i, j]
      αᵢⱼ = (
        m.ξx * (m.ξxᵢ₊½ - m.ξxᵢ₋½) +
        m.ξy * (m.ξyᵢ₊½ - m.ξyᵢ₋½) +
        m.ηx * (m.ξxⱼ₊½ - m.ξxⱼ₋½) +
        m.ηy * (m.ξyⱼ₊½ - m.ξyⱼ₋½)
      )

      βᵢⱼ = (
        m.ξx * (m.ηxᵢ₊½ - m.ηxᵢ₋½) +
        m.ξy * (m.ηyᵢ₊½ - m.ηyᵢ₋½) +
        m.ηx * (m.ηxⱼ₊½ - m.ηxⱼ₋½) +
        m.ηy * (m.ηyⱼ₊½ - m.ηyⱼ₋½)
      )

      #! format: off
      uⁿ⁺¹[i, j] =
        Δt * (
          (m.ξx^2 + m.ξy^2) * (aᵢ₊½ * (u[i + 1, j] - u[i, j]) - aᵢ₋½ * (u[i, j] - u[i - 1, j])) +
          (m.ηx^2 + m.ηy^2) * (aⱼ₊½ * (u[i, j + 1] - u[i, j]) - aⱼ₋½ * (u[i, j] - u[i, j - 1])) +
          0.25(m.ξx * m.ηx + m.ξy * m.ηy) * (aᵢ₊₁ⱼ * (u[i + 1, j + 1] - u[i + 1, j - 1]) - aᵢ₋₁ⱼ * (u[i - 1, j + 1] - u[i - 1, j - 1])) +
          0.25(m.ξx * m.ηx + m.ξy * m.ηy) * (aᵢⱼ₊₁ * (u[i + 1, j + 1] - u[i - 1, j + 1]) - aᵢⱼ₋₁ * (u[i + 1, j - 1] - u[i - 1, j - 1])) +
          0.5aᵢⱼ*αᵢⱼ * (u[i + 1, j] - u[i - 1, j]) + 0.5aᵢⱼ*βᵢⱼ * (u[i, j + 1] - u[i, j - 1]) +
          solver.source_term[i, j] + (u[i, j] / Δt)
        )
      #! format: on
      uⁿ⁺¹[i, j] = cutoff(uⁿ⁺¹[i, j])
    end
  end

  for j in jlo:jhi
    for i in ilo:ihi
      u[i, j] = uⁿ⁺¹[i, j]
    end
  end

  for j in jlo:jhi
    for i in ilo:ihi
      if !isfinite(u[i, j]) || u[i, j] < 0
        @show u[i, j]
        # @show qⁿ⁺¹[i, j]
        # @show pⁿ⁺¹[i, j]
        error("Invalid value for u $(u[i,j]) at $((i,j))")
      end
    end
  end

  # N = (ihi - ilo + 1) * (jhi - jlo + 1)
  # L₂ = sqrt(L₂ / N)

  return 0.0, 0.0
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

function compute_metrics_fd(mesh, (i, j))
  xᵢⱼ, yᵢⱼ = coord(mesh, (i + 0.5, j + 0.5))
  xᵢ₊₁ⱼ, yᵢ₊₁ⱼ = coord(mesh, (i + 1.5, j))
  xᵢ₊₁ⱼ₊₁, yᵢ₊₁ⱼ₊₁ = coord(mesh, (i + 1.5, j + 1.5))
  xᵢ₊₁ⱼ₋₁, yᵢ₊₁ⱼ₋₁ = coord(mesh, (i + 1.5, j - 1.5))
  xᵢ₋₁ⱼ₋₁, yᵢ₋₁ⱼ₋₁ = coord(mesh, (i - 1.5, j - 1.5))
  xᵢ₋₁ⱼ₊₁, yᵢ₋₁ⱼ₊₁ = coord(mesh, (i - 1.5, j + 1.5))
  xᵢ₋₁ⱼ, yᵢ₋₁ⱼ = coord(mesh, (i - 1.5, j + 0.5))
  xᵢⱼ₊₁, yᵢⱼ₊₁ = coord(mesh, (i + 0.5, j + 1.5))
  xᵢⱼ₋₁, yᵢⱼ₋₁ = coord(mesh, (i + 0.5, j - 1.5))
  # xᵢⱼ, yᵢⱼ =         coord(mesh, (i, j))
  # xᵢ₊₁ⱼ, yᵢ₊₁ⱼ =     coord(mesh, (i + 1, j))
  # xᵢ₊₁ⱼ₊₁, yᵢ₊₁ⱼ₊₁ = coord(mesh, (i + 1, j + 1))
  # xᵢ₊₁ⱼ₋₁, yᵢ₊₁ⱼ₋₁ = coord(mesh, (i + 1, j - 1))
  # xᵢ₋₁ⱼ₋₁, yᵢ₋₁ⱼ₋₁ = coord(mesh, (i - 1, j - 1))
  # xᵢ₋₁ⱼ₊₁, yᵢ₋₁ⱼ₊₁ = coord(mesh, (i - 1, j + 1))
  # xᵢ₋₁ⱼ, yᵢ₋₁ⱼ =     coord(mesh, (i - 1, j))
  # xᵢⱼ₊₁, yᵢⱼ₊₁ =     coord(mesh, (i, j + 1))
  # xᵢⱼ₋₁, yᵢⱼ₋₁ =     coord(mesh, (i, j - 1))

  J = 0.25 * ((xᵢ₊₁ⱼ - xᵢ₋₁ⱼ) * (yᵢⱼ₊₁ - yᵢⱼ₋₁) - (xᵢⱼ₊₁ - xᵢⱼ₋₁) * (yᵢ₊₁ⱼ - yᵢ₋₁ⱼ))
  Jᵢ₊½ =
    0.25 * (
      (xᵢ₊₁ⱼ - xᵢⱼ) * (yᵢⱼ₊₁ - yᵢⱼ₋₁ + yᵢ₊₁ⱼ₊₁ - yᵢ₊₁ⱼ₋₁) -
      (yᵢ₊₁ⱼ - yᵢⱼ) * (xᵢⱼ₊₁ - xᵢⱼ₋₁ + xᵢ₊₁ⱼ₊₁ - xᵢ₊₁ⱼ₋₁)
    )
  Jᵢ₋½ =
    0.25 * (
      (xᵢⱼ - xᵢ₋₁ⱼ) * (yᵢⱼ₊₁ - yᵢⱼ₋₁ + yᵢ₋₁ⱼ₊₁ - yᵢ₋₁ⱼ₋₁) -
      (yᵢⱼ - yᵢ₋₁ⱼ) * (xᵢⱼ₊₁ - xᵢⱼ₋₁ + xᵢ₋₁ⱼ₊₁ - xᵢ₋₁ⱼ₋₁)
    )
  Jⱼ₊½ =
    0.25 * (
      (yᵢⱼ₊₁ - yᵢⱼ) * (xᵢ₊₁ⱼ - xᵢ₋₁ⱼ + xᵢ₊₁ⱼ₊₁ - xᵢ₋₁ⱼ₊₁) -
      (xᵢⱼ₊₁ - xᵢⱼ) * (yᵢ₊₁ⱼ - yᵢ₋₁ⱼ + yᵢ₊₁ⱼ₊₁ - yᵢ₋₁ⱼ₊₁)
    )
  Jⱼ₋½ =
    0.25 * (
      (yᵢⱼ - yᵢⱼ₋₁) * (xᵢ₊₁ⱼ - xᵢ₋₁ⱼ + xᵢ₊₁ⱼ₋₁ - xᵢ₋₁ⱼ₋₁) -
      (xᵢⱼ - xᵢⱼ₋₁) * (yᵢ₊₁ⱼ - yᵢ₋₁ⱼ + yᵢ₊₁ⱼ₋₁ - yᵢ₋₁ⱼ₋₁)
    )

  ξx = (yᵢⱼ₊₁ - yᵢⱼ₋₁) / (2J)
  ξy = -(xᵢⱼ₊₁ - xᵢⱼ₋₁) / (2J)
  ηx = -(yᵢ₊₁ⱼ - yᵢ₋₁ⱼ) / (2J)
  ηy = (xᵢ₊₁ⱼ - xᵢ₋₁ⱼ) / (2J)

  ξxᵢ₊½ = (yᵢⱼ₊₁ - yᵢⱼ₋₁ + yᵢ₊₁ⱼ₊₁ - yᵢ₊₁ⱼ₋₁) / (4Jᵢ₊½)
  ξxᵢ₋½ = (yᵢⱼ₊₁ - yᵢⱼ₋₁ + yᵢ₋₁ⱼ₊₁ - yᵢ₋₁ⱼ₋₁) / (4Jᵢ₋½)
  ξyᵢ₊½ = (xᵢⱼ₊₁ - xᵢⱼ₋₁ + xᵢ₊₁ⱼ₊₁ - xᵢ₊₁ⱼ₋₁) / (-4Jᵢ₊½)
  ξyᵢ₋½ = (xᵢⱼ₊₁ - xᵢⱼ₋₁ + xᵢ₋₁ⱼ₊₁ - xᵢ₋₁ⱼ₋₁) / (-4Jᵢ₋½)

  ηxⱼ₊½ = (yᵢ₊₁ⱼ - yᵢ₋₁ⱼ + yᵢ₊₁ⱼ₊₁ - yᵢ₋₁ⱼ₊₁) / (-4Jⱼ₊½)
  ηxⱼ₋½ = (yᵢ₊₁ⱼ - yᵢ₋₁ⱼ + yᵢ₊₁ⱼ₋₁ - yᵢ₋₁ⱼ₋₁) / (-4Jⱼ₋½)

  ηyⱼ₊½ = (xᵢ₊₁ⱼ - xᵢ₋₁ⱼ + xᵢ₊₁ⱼ₊₁ - xᵢ₋₁ⱼ₊₁) / (4Jⱼ₊½)
  ηyⱼ₋½ = (xᵢ₊₁ⱼ - xᵢ₋₁ⱼ + xᵢ₊₁ⱼ₋₁ - xᵢ₋₁ⱼ₋₁) / (4Jⱼ₋½)

  ξxⱼ₊½ = (yᵢⱼ₊₁ - yᵢⱼ) / Jⱼ₊½
  ξxⱼ₋½ = (yᵢⱼ - yᵢⱼ₋₁) / Jⱼ₋½

  ξyⱼ₊½ = -(xᵢⱼ₊₁ - xᵢⱼ) / Jⱼ₊½
  ξyⱼ₋½ = -(xᵢⱼ - xᵢⱼ₋₁) / Jⱼ₋½

  ηxᵢ₊½ = -(yᵢ₊₁ⱼ - yᵢⱼ) / Jᵢ₊½
  ηxᵢ₋½ = -(yᵢⱼ - yᵢ₋₁ⱼ) / Jᵢ₋½

  ηyᵢ₊½ = (xᵢ₊₁ⱼ - xᵢⱼ) / Jᵢ₊½
  ηyᵢ₋½ = (xᵢⱼ - xᵢ₋₁ⱼ) / Jᵢ₋½

  cell_metrics = (;
    # J,
    ξx,
    ξy,
    ηx,
    ηy,
    ξxᵢ₋½,
    ξyᵢ₋½,
    ηxᵢ₋½,
    ηyᵢ₋½,
    ξxⱼ₋½,
    ξyⱼ₋½,
    ηxⱼ₋½,
    ηyⱼ₋½,
    ξxᵢ₊½,
    ξyᵢ₊½,
    ηxᵢ₊½,
    ηyᵢ₊½,
    ξxⱼ₊½,
    ξyⱼ₊½,
    ηxⱼ₊½,
    ηyⱼ₊½,
  )

  return cell_metrics
end

end
