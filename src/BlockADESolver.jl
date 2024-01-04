using Plots

# TODO: make a linear and non-linear version based on κ or a
struct BlockADESolver{BHA,T,N,EM,F,BC}
  u::NTuple{3,BHA}
  u_edge::NTuple{2,BHA}
  ϵ::BHA # change in the solution between stages
  qⁿ⁺¹::BHA
  pⁿ⁺¹::BHA
  J::Array{T,N} # cell-centered Jacobian
  metrics::Array{EM,N}
  a::Array{T,N} # cell-centered diffusivity
  source_term::Array{T,N} # cell-centered source term
  mean_func::F
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
function BlockADESolver(
  mesh::CurvilinearGrid2D,
  bcs,
  nblocks=nthreads(),
  form=:conservative,
  mean_func=arithmetic_mean,
  T=Float64,
)
  celldims = cellsize_withhalo(mesh)
  celldims_nohalo = cellsize(mesh)

  overlap = mesh.nhalo

  u = (
    BlockHaloArray(celldims_nohalo, overlap, nblocks; T=T),
    BlockHaloArray(celldims_nohalo, overlap, nblocks; T=T),
    BlockHaloArray(celldims_nohalo, overlap, nblocks; T=T),
  )
  u_edge = (
    BlockHaloArray(celldims_nohalo, overlap, nblocks; T=T),
    BlockHaloArray(celldims_nohalo, overlap, nblocks; T=T),
  )
  ϵ = BlockHaloArray(celldims_nohalo, overlap, nblocks; T=T)
  qⁿ⁺¹ = BlockHaloArray(celldims_nohalo, overlap, nblocks; T=T)
  pⁿ⁺¹ = BlockHaloArray(celldims_nohalo, overlap, nblocks; T=T)

  # qⁿ⁺¹ = zeros(T, celldims)
  # pⁿ⁺¹ = zeros(T, celldims)
  J = zeros(T, celldims)
  # ϵ = zeros(T, celldims)

  # @unpack ilo, ihi, jlo, jhi = mesh.limits

  # block_ranges = Vector{CartesianIndices}(undef, nblocks)
  # for tid in 1:nblocks
  #   b_ilo, b_ihi, b_jlo, b_jhi = tile_indices_2d(celldims; ntiles=nblocks, id=tid)

  #   # clip due to halo cells
  #   _ilo = max(b_ilo, ilo)
  #   _jlo = max(b_jlo, jlo)
  #   _ihi = min(b_ihi, ihi)
  #   _jhi = min(b_jhi, jhi)

  #   CI = CartesianIndices((_ilo:_ihi, _jlo:_jhi))
  #   block_ranges[tid] = CI
  # end

  # if form === :conservative
  #   conservative = true
  #   metric_type = typeof(_conservative_metrics_2d(mesh, 1, 1))
  # else
  conservative = false
  metric_type = typeof(_non_conservative_metrics_2d(mesh, 1, 1))
  # end

  # u_edge_stage = ntuple(i -> zeros(T, celldims), 2)
  edge_metrics = Array{metric_type,2}(undef, celldims)

  diffusivity = zeros(T, celldims)
  source_term = zeros(T, celldims)
  # limits = mesh.limits

  solver = BlockADESolver(
    u,
    u_edge,
    ϵ,
    qⁿ⁺¹,
    pⁿ⁺¹,
    J,
    edge_metrics,
    diffusivity,
    source_term,
    mean_func,
    bcs,
    overlap,
    conservative,
  )
  update_mesh_metrics!(solver, mesh)

  return solver
end

# """Update the mesh metrics. Only do this whenever the mesh moves"""
# function update_mesh_metrics!(solver, mesh::CurvilinearGrid2D)
#   @unpack ilo, ihi, jlo, jhi = mesh.limits

#   if solver.conservative
#     @inline for j in jlo:jhi
#       for i in ilo:ihi
#         solver.J[i, j] = jacobian(mesh, (i, j))
#         solver.metrics[i, j] = _conservative_metrics_2d(mesh, i, j)
#       end
#     end
#   else
#     @inline for j in jlo:jhi
#       for i in ilo:ihi
#         solver.J[i, j] = jacobian(mesh, (i, j))
#         solver.metrics[i, j] = _non_conservative_metrics_2d(mesh, i, j)
#       end
#     end
#   end

#   return nothing
# end

# @inline function _non_conservative_metrics_2d(mesh, i, j)
#   # note, metrics(mesh, (i,j)) uses node-based indexing, and here we're making
#   # a tuple of metrics that uses cell-based indexing, thus the weird 1/2 offsets
#   metricsᵢⱼ = cell_metrics(mesh, (i, j))
#   metricsᵢ₊½ = cell_metrics(mesh, (i + 1 / 2, j))
#   metricsⱼ₊½ = cell_metrics(mesh, (i, j + 1 / 2))
#   metricsᵢ₋½ = cell_metrics(mesh, (i - 1 / 2, j))
#   metricsⱼ₋½ = cell_metrics(mesh, (i, j - 1 / 2))
#   # metricsᵢⱼ = metrics(mesh, (i + 1 / 2, j + 1 / 2))
#   # metricsᵢ₋½ = metrics(mesh, (i, j + 1 / 2))
#   # metricsⱼ₋½ = metrics(mesh, (i + 1 / 2, j))
#   # metricsᵢ₊½ = metrics(mesh, (i + 1, j + 1 / 2))
#   # metricsⱼ₊½ = metrics(mesh, (i + 1 / 2, j + 1))

#   return (
#     ξx=metricsᵢⱼ.ξx,
#     ξy=metricsᵢⱼ.ξy,
#     ηx=metricsᵢⱼ.ηx,
#     ηy=metricsᵢⱼ.ηy,
#     ξxᵢ₋½=metricsᵢ₋½.ξx,
#     ξyᵢ₋½=metricsᵢ₋½.ξy,
#     ηxᵢ₋½=metricsᵢ₋½.ηx,
#     ηyᵢ₋½=metricsᵢ₋½.ηy,
#     ξxⱼ₋½=metricsⱼ₋½.ξx,
#     ξyⱼ₋½=metricsⱼ₋½.ξy,
#     ηxⱼ₋½=metricsⱼ₋½.ηx,
#     ηyⱼ₋½=metricsⱼ₋½.ηy,
#     ξxᵢ₊½=metricsᵢ₊½.ξx,
#     ξyᵢ₊½=metricsᵢ₊½.ξy,
#     ηxᵢ₊½=metricsᵢ₊½.ηx,
#     ηyᵢ₊½=metricsᵢ₊½.ηy,
#     ξxⱼ₊½=metricsⱼ₊½.ξx,
#     ξyⱼ₊½=metricsⱼ₊½.ξy,
#     ηxⱼ₊½=metricsⱼ₊½.ηx,
#     ηyⱼ₊½=metricsⱼ₊½.ηy,
#   )
# end

# @inline cutoff(a) = (0.5(abs(a) + a))

# @inline function _conservative_metrics_2d(mesh, i, j)
#   metrics_i_plus_half = metrics_with_jacobian(mesh, (i + 1, j))
#   metrics_i_minus_half = metrics_with_jacobian(mesh, (i, j))
#   metrics_j_plus_half = metrics_with_jacobian(mesh, (i, j + 1))
#   metrics_j_minus_half = metrics_with_jacobian(mesh, (i, j))
#   # metrics_i_plus_half = metrics_with_jacobian(mesh, (i + 1 / 2, j))
#   # metrics_i_minus_half = metrics_with_jacobian(mesh, (i - 1 / 2, j))
#   # metrics_j_plus_half = metrics_with_jacobian(mesh, (i, j + 1 / 2))
#   # metrics_j_minus_half = metrics_with_jacobian(mesh, (i, j - 1 / 2))

#   return (
#     Jᵢ₊½=metrics_i_plus_half.J,
#     Jξx_ᵢ₊½=metrics_i_plus_half.ξx * metrics_i_plus_half.J,
#     Jξy_ᵢ₊½=metrics_i_plus_half.ξy * metrics_i_plus_half.J,
#     Jηx_ᵢ₊½=metrics_i_plus_half.ηx * metrics_i_plus_half.J,
#     Jηy_ᵢ₊½=metrics_i_plus_half.ηy * metrics_i_plus_half.J,
#     Jᵢ₋½=metrics_i_minus_half.J,
#     Jξx_ᵢ₋½=metrics_i_minus_half.ξx * metrics_i_minus_half.J,
#     Jξy_ᵢ₋½=metrics_i_minus_half.ξy * metrics_i_minus_half.J,
#     Jηx_ᵢ₋½=metrics_i_minus_half.ηx * metrics_i_minus_half.J,
#     Jηy_ᵢ₋½=metrics_i_minus_half.ηy * metrics_i_minus_half.J,
#     Jⱼ₊½=metrics_j_plus_half.J,
#     Jξx_ⱼ₊½=metrics_j_plus_half.ξx * metrics_j_plus_half.J,
#     Jξy_ⱼ₊½=metrics_j_plus_half.ξy * metrics_j_plus_half.J,
#     Jηx_ⱼ₊½=metrics_j_plus_half.ηx * metrics_j_plus_half.J,
#     Jηy_ⱼ₊½=metrics_j_plus_half.ηy * metrics_j_plus_half.J,
#     Jⱼ₋½=metrics_j_minus_half.J,
#     Jξx_ⱼ₋½=metrics_j_minus_half.ξx * metrics_j_minus_half.J,
#     Jξy_ⱼ₋½=metrics_j_minus_half.ξy * metrics_j_minus_half.J,
#     Jηx_ⱼ₋½=metrics_j_minus_half.ηx * metrics_j_minus_half.J,
#     Jηy_ⱼ₋½=metrics_j_minus_half.ηy * metrics_j_minus_half.J,
#   )
# end

# """
# When using the diffusion solver to solve the heat equation, this can be used to
# update the thermal conductivity of the mesh at each cell-center.

# # Arguments
#  - `solver::BlockADESolver`: The solver type
#  - `T::Array`: Temperature
#  - `ρ::Array`: Density
#  - `κ::Function`: Function to determine thermal conductivity, e.g. κ(ρ,T) = κ0 * ρ * T^(5/2)
#  - `cₚ::Real`: Heat capacity
# """
# function update_conductivity!(
#   solver::BlockADESolver, T::Array, ρ::Array, κ::Function, cₚ::Real
# )
#   @inline for idx in eachindex(T)
#     rho = ρ[idx]
#     kappa = κ(rho, T[idx])
#     solver.a[idx] = kappa / (rho * cₚ)
#   end

#   return nothing
# end

# function update_diffusivity(BlockADESolver::solver, κ)
#   @inline for idx in eachindex(solver.diffusivity)
#   end
# end

# """
# # Arguments
#  - α: Diffusion coefficient
# """

# solve!(solver::BlockADESolver, mesh, u, Δt) = solve_nc_nonlinear!(solver, mesh, u, Δt)
# solve!(solver::BlockADESolver, mesh, u, Δt) = solve_nc_nonlinear_explicit!(solver, mesh, u, Δt)

function solve_conservative!(solver::BlockADESolver, mesh, u, Δt)
  @unpack ilo, ihi, jlo, jhi = solver.limits

  α = solver.a

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

function solve_innersweep_sync!(
  solver::BlockADESolver, mesh, u::AbstractArray, Δt::Real, maxcycles=Inf, tol=1e-6
)
  u⁽⁰⁾ = solver.u[1]
  u⁽¹⁾ = solver.u[2]
  u⁽²⁾ = solver.u[3]

  n_blocks = nblocks(u⁽⁰⁾)
  cycle = 0

  ilo, ihi, jlo, jhi = mesh.limits
  u_dom = @views u[ilo:ihi, jlo:jhi]
  applybc!(u, solver.bcs, solver.nhalo)

  copy!(u⁽⁰⁾, u_dom) # save the initial state - we revert to this each cycle
  copy!(u⁽¹⁾, u_dom)

  # updatehalo!(u⁽⁰⁾) # populate the halo cells
  updatehalo!(u⁽¹⁾) # populate the halo cells

  # 1st solve
  nc_solve(solver, u⁽¹⁾, Δt)
  updatehalo!(u⁽¹⁾)

  copy!(u⁽²⁾, u⁽¹⁾)
  while true
    cycle += 1

    # populate the halo cells of all the neighbors
    # so the boundaries get updated
    updatehalo!(u⁽²⁾)

    # Reset the domain values. This ignores the halo cells,
    # which is what we want. The newly updated halo cells will
    # be used in the next solve
    copy_domain!(u⁽²⁾, u⁽⁰⁾)

    # solve with the updated boundary values
    nc_solve(solver, u⁽²⁾, Δt)

    L₂ = compare_solution_norm(u⁽²⁾, u⁽¹⁾)
    is_converged = (L₂ <= tol)

    if is_converged
      copy!(u_dom, u⁽²⁾) # copy solution back to u
      # error("done")
      return L₂, cycle, is_converged
    else
      # we haven't converged yet, so store this cycle's solution
      # as the new "previous" solution
      copy!(u⁽¹⁾, u⁽²⁾)
    end

    if cycle == maxcycles
      @warn "The BlockADESolver is hitting the maxcycles limit. The solution is likely not converged!"
      copy!(u_dom, u⁽¹⁾) # copy solution back to u
      return L₂, cycle, is_converged
    end
  end # cycle loop

  return nothing
end

function solve!(
  solver::BlockADESolver, mesh, u::AbstractArray, Δt::Real, maxcycles=Inf, tol=1e-8
)
  u⁽⁰⁾ = solver.u[1]
  u⁽¹⁾ = solver.u[2]
  u⁽²⁾ = solver.u[3]

  n_blocks = nblocks(u⁽⁰⁾)
  cycle = 0

  ilo, ihi, jlo, jhi = mesh.limits
  u_dom = @views u[ilo:ihi, jlo:jhi]
  applybc!(u, solver.bcs, solver.nhalo)

  copy!(u⁽⁰⁾, u_dom) # save the initial state - we revert to this each cycle
  copy!(u⁽¹⁾, u_dom)

  # updatehalo!(u⁽⁰⁾) # populate the halo cells
  updatehalo!(u⁽¹⁾) # populate the halo cells
  bid = 3

  # 1st solve
  @sync for blockid in 1:n_blocks
    # u⁽¹⁾ is now the 1st guess at the solution.
    # edges are out of sync still -- this is why we need to cycle a few times
    @tspawnat blockid solve_block!(solver, u⁽¹⁾, Δt, blockid)
  end
  updatehalo!(u⁽¹⁾)

  # p = heatmap(u⁽¹⁾[bid]; title="u⁽¹⁾")
  # display(p)
  copy!(u⁽²⁾, u⁽¹⁾)

  L2_sol = @MVector zeros(n_blocks)
  while true
    cycle += 1

    # populate the halo cells of all the neighbors
    # so the boundaries get updated
    updatehalo!(u⁽²⁾)

    # Reset the domain values. This ignores the halo cells,
    # which is what we want. The newly updated halo cells will
    # be used in the next solve
    copy_domain!(u⁽²⁾, u⁽⁰⁾)

    # p = heatmap(u⁽²⁾[bid]; title="u⁽²⁾")
    # display(p)
    # p3 = heatmap(u⁽²⁾[bid] .- u⁽⁰⁾[bid]; title="u⁽²⁾ .- u⁽⁰⁾")
    # display(p3)

    # solve with the updated boundary values
    @sync for blockid in 1:n_blocks
      @tspawnat blockid begin
        # solve the block now with updated boundary values
        # in the halo cells
        # u⁽²⁾[blockid][end, :] .= 4.0
        L2_block = solve_block!(solver, u⁽²⁾, Δt, blockid)
        L2_sol[blockid] = L2_block
      end
    end
    # @show maximum(L2_sol)

    # p1 = heatmap(u⁽¹⁾[bid]; title="u⁽¹⁾post")
    # display(p1)
    # p2 = heatmap(u⁽²⁾[bid]; title="u⁽²⁾post")
    # display(p2)

    # p3 = heatmap(u⁽¹⁾[bid] .- u⁽²⁾[bid]; title="u⁽¹⁾ .- u⁽²⁾")
    # display(p3)

    L₂ = compare_solution_norm(u⁽²⁾, u⁽¹⁾)
    # L₂ = compare_solution_norm(u⁽²⁾, u⁽⁰⁾)
    is_converged = (L₂ <= tol)
    # @show L₂, cycle, is_converged

    if is_converged
      copy!(u_dom, u⁽²⁾) # copy solution back to u
      # error("done")
      return L₂, cycle, is_converged
    else
      # we haven't converged yet, so store this cycle's solution
      # as the new "previous" solution
      copy!(u⁽¹⁾, u⁽²⁾)
    end

    if cycle == maxcycles
      @warn "The BlockADESolver is hitting the maxcycles limit. The solution is likely not converged!"
      copy!(u_dom, u⁽¹⁾) # copy solution back to u
      return L₂, cycle, is_converged
    end
  end # cycle loop

  return nothing
end

function solve_orig_edge!(
  solver::BlockADESolver, mesh, u::AbstractArray, Δt::Real, maxcycles=5, tol=1e-5
)
  u⁽⁰⁾ = solver.u[1]
  u⁽¹⁾ = solver.u[2]
  u_edge⁽¹⁾ = solver.u_edge[1]
  u_edge⁽²⁾ = solver.u_edge[2]
  ilo, ihi, jlo, jhi = mesh.limits
  u_dom = @views u[ilo:ihi, jlo:jhi]
  applybc!(u, solver.bcs, solver.nhalo)

  copy!(u⁽⁰⁾, u_dom)
  updatehalo!(u⁽⁰⁾) # populate the halo cells

  cycle = 0
  n_blocks = nblocks(u⁽⁰⁾)

  # TODO: I suspect something is amiss with the edges... I'm getting L2 of 0!
  # 1st solve
  @sync for blockid in 1:n_blocks
    @tspawnat blockid begin
      fill!(u_edge⁽¹⁾[blockid], 0)
      fill!(u_edge⁽²⁾[blockid], 0)
      solve_block!(solver, u⁽¹⁾, Δt, blockid)

      # save the boundary/edge values in u_edge⁽¹⁾
      # we use this to keep track of how much the answer has changed based on the boundaries
      update_block_edges!(u⁽¹⁾, u_edge⁽¹⁾, blockid, 0)
    end
  end

  while true
    cycle += 1

    # use u⁽¹⁾ to populate the halo cells of all the neighbors
    updatehalo!(u⁽¹⁾)

    # reset the domain values
    # this ignores the halo cells, which is what we want. The
    # newly updated halo cells will be used in the next solve
    copy_domain!(u⁽¹⁾, u⁽⁰⁾)

    # solve with the updated boundary values
    @sync for blockid in 1:n_blocks
      @tspawnat blockid begin
        # fill!(u_edge⁽²⁾[blockid], 0)

        # solve the block (now we have updated boundary values)
        solve_block!(solver, u⁽¹⁾, Δt, blockid)

        # copy the boundary values to u_edge⁽²⁾, so we can keep
        # track of convergence by comparing u_edge⁽²⁾ and u_edge⁽¹⁾
        update_block_edges!(u⁽¹⁾, u_edge⁽²⁾, blockid, cycle)
      end
    end

    # find L2 norm of u_edge⁽²⁾ - u_edge⁽¹⁾
    L₂ = check_edge_convergence(solver)
    is_converged = (L₂ <= tol)
    @show L₂, cycle, is_converged

    if is_converged
      copy!(u_dom, u⁽¹⁾) # copy solution back to u
      error("done")
      return L₂, cycle, is_converged
    else
      copy!(u_edge⁽¹⁾, u_edge⁽²⁾)
    end

    if cycle == maxcycles
      @warn "The BlockADESolver is hitting the maxcycles limit. The solution is likely not converged!"
      copy!(u_dom, u⁽¹⁾) # copy solution back to u
      return L₂, cycle, is_converged
    end
  end # cycle loop

  return nothing
end

function solve_block!(solver, u⁽¹⁾, Δt, blockid)
  if solver.conservative
    solve_conservative_block!(solver, u⁽¹⁾, Δt, blockid)
  else
    solve_nc_nonlinear_block!(solver, u⁽¹⁾, Δt, blockid)
  end
end

# update the edge values for a particular block
function update_block_edges!(u, u_edge, blockid, cycle)
  u_block = u[blockid]
  edge_block = u_edge[blockid]

  ilo, ihi, jlo, jhi = u.loop_limits[blockid]

  # if blockid == 3
  #   p1 = heatmap(edge_block; title="Block $(blockid) before, cycle: $cycle")
  #   display(p1)
  # end

  for j in jlo:jhi
    for i in ilo:ihi
      on_edge = (i == ilo || i == ihi || j == jlo || j == jhi)

      if on_edge
        edge_block[i, j] = u_block[i, j]
      end
    end
  end

  # if blockid == 3
  #   p2 = heatmap(edge_block; title="Block $(blockid) after, cycle: $cycle")
  #   display(p2)
  # end
  return nothing
end

# find the maximum L₂ norm of the change in edge values
function check_edge_convergence(solver)
  nb = nblocks(solver.u[1])
  L2 = @MVector zeros(nb)

  @sync for blockid in 1:nb
    @tspawnat blockid begin
      L₂_block = check_edge_convergence_block(solver, blockid)
      L2[blockid] = L₂_block
    end
  end

  L2max = maximum(L2)
  return L2max
end

# find out how much each block's edge values have changed
function check_edge_convergence_block(solver, blockid)
  old = solver.u_edge[1][blockid]
  new = solver.u_edge[2][blockid]
  ϵ = solver.ϵ[blockid]

  ilo, ihi, jlo, jhi = solver.ϵ.loop_limits[blockid]
  L₂denom = 0.0
  L₂numer = 0.0
  for j in jlo:jhi
    for i in ilo:ihi
      on_edge = (i == ilo || i == ihi || j == jlo || j == jhi)

      if on_edge
        L₂numer += abs2(old[i, j] - new[i, j])
        L₂denom += abs2(old[i, j])
        ϵ[i, j] = abs(old[i, j] - new[i, j])
      end
    end
  end

  # if blockid == 3
  #   p = heatmap(ϵ; title="ϵ")
  #   display(p)
  # end

  if !isfinite(L₂denom) || iszero(L₂denom)
    L2norm = -Inf
  else
    L2norm = sqrt(L₂numer) / sqrt(L₂denom)
  end

  return L2norm
end

function compare_solution_norm(u2, u1)
  nb = nblocks(u2)
  L2 = @MVector zeros(nb)

  @sync for blockid in 1:nb
    @tspawnat blockid begin
      L₂_block = compare_solution_norm_block(u2, u1, blockid)
      L2[blockid] = L₂_block
    end
  end

  # @show L2
  L2max = maximum(L2)
  return L2max
end

function compare_solution_norm_block(u2, u1, blockid)
  d2 = domainview(u2, blockid)
  d1 = domainview(u1, blockid)
  L₂denom = 0.0
  L₂numer = 0.0

  @inline for idx in eachindex(d2)
    L₂numer += abs2(d2[idx] - d1[idx])
    L₂denom += abs2(d2[idx])
  end

  if !isfinite(L₂denom) || iszero(L₂denom)
    L2norm = -Inf
  else
    L2norm = sqrt(L₂numer) / sqrt(L₂denom)
  end

  return L2norm
end

# Solve on the given block using the non-conservative form
function solve_nc_nonlinear_block!(solver::BlockADESolver, u0, Δt, blockid)
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

  ilo, ihi, jlo, jhi = u0.loop_limits[blockid]

  nhalo = solver.nhalo
  overlap = nhalo - 1
  blk_ilo = ilo - overlap
  blk_ihi = ihi + overlap
  blk_jlo = jlo - overlap
  blk_jhi = jhi + overlap

  global_ranges = u0.global_blockranges[blockid]
  # @show global_ranges
  ilo_g = first(global_ranges[1]) - overlap
  jlo_g = first(global_ranges[2]) - overlap
  ihi_g = last(global_ranges[1]) + overlap
  jhi_g = last(global_ranges[2]) + overlap

  # globalCI = CartesianIndices(u0.global_blockranges[blockid])
  globalCI = CartesianIndices((ilo_g:ihi_g, jlo_g:jhi_g))
  blockCI = CartesianIndices((blk_ilo:blk_ihi, blk_jlo:blk_jhi))
  # @show blockCI, globalCI
  @assert length(globalCI) == length(blockCI)
  # error("done!")

  α = solver.a
  # applybc!(u, solver.bcs, solver.nhalo) # update the ghost cell temperatures

  #------------------------------------------------------
  # Forward Sweep
  #------------------------------------------------------
  # anything at i-1 or j-1 is from time level n+1, since we already computed it

  # make alias for code readibilty
  pⁿ⁺¹ = solver.pⁿ⁺¹[blockid]
  qⁿ⁺¹ = solver.qⁿ⁺¹[blockid]
  u = u0[blockid]
  @inline for idx in eachindex(u)
    pⁿ⁺¹[idx] = u[idx]
    qⁿ⁺¹[idx] = u[idx]
  end
  pⁿ = @views pⁿ⁺¹
  qⁿ = @views qⁿ⁺¹

  for (gidx, bidx) in zip(
    globalCI, # global indices
    blockCI,  # block indices
  )
    # @show bidx, gidx
    bi, bj = Tuple(bidx) # block i, j
    gi, gj = Tuple(gidx) .+ solver.nhalo # global i, j
    aᵢ₊½ = solver.mean_func(α[gi, gj], α[gi + 1, gj])
    aᵢ₋½ = solver.mean_func(α[gi, gj], α[gi - 1, gj])
    aⱼ₊½ = solver.mean_func(α[gi, gj], α[gi, gj + 1])
    aⱼ₋½ = solver.mean_func(α[gi, gj], α[gi, gj - 1])

    aᵢⱼ = α[gi, gj]
    aᵢ₊₁ⱼ = α[gi + 1, gj]
    aᵢ₋₁ⱼ = α[gi - 1, gj]
    aᵢⱼ₊₁ = α[gi, gj + 1]
    aᵢⱼ₋₁ = α[gi, gj - 1]

    m = solver.metrics[gi, gj]
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

    pⁿ⁺¹[bi, bj] =
      (
        # orthogonal terms
        (m.ξx^2 + m.ξy^2) *
        (aᵢ₊½ * (pⁿ[bi + 1, bj] - pⁿ[bi, bj]) + aᵢ₋½ * pⁿ⁺¹[bi - 1, bj]) +
        (m.ηx^2 + m.ηy^2) *
        (aⱼ₊½ * (pⁿ[bi, bj + 1] - pⁿ[bi, bj]) + aⱼ₋½ * pⁿ⁺¹[bi, bj - 1]) +
        # non-orthogonal terms
        0.25(m.ξx * m.ηx + m.ξy * m.ηy) * (
          aᵢ₊₁ⱼ * (pⁿ[bi + 1, bj + 1] - pⁿ[bi + 1, bj - 1]) -
          aᵢ₋₁ⱼ * (pⁿ[bi - 1, bj + 1] - pⁿ[bi - 1, bj - 1])
        ) +
        0.25(m.ξx * m.ηx + m.ξy * m.ηy) * (
          aᵢⱼ₊₁ * (pⁿ[bi + 1, bj + 1] - pⁿ[bi - 1, bj + 1]) -
          aᵢⱼ₋₁ * (pⁿ[bi + 1, bj - 1] - pⁿ[bi - 1, bj - 1])
        ) +
        # geometric terms
        0.5aᵢⱼ * αᵢⱼ * (pⁿ[bi + 1, bj] - pⁿ⁺¹[bi - 1, bj]) +
        0.5aᵢⱼ * βᵢⱼ * (pⁿ[bi, bj + 1] - pⁿ⁺¹[bi, bj - 1]) +
        # source and remaining terms
        solver.source_term[gi, gj] +
        (pⁿ[bi, bj] / Δt)
      ) / ((1 / Δt) + (aᵢ₋½ * (m.ξx^2 + m.ξy^2) + aⱼ₋½ * (m.ηx^2 + m.ηy^2)))

    pⁿ⁺¹[bi, bj] = cutoff(pⁿ⁺¹[bi, bj])
  end

  #------------------------------------------------------
  # Reverse Sweep
  #------------------------------------------------------
  # anything at i+1 or j+1 is from time level n+1, since we already computed it
  @inline for (gidx, bidx) in zip(
    Iterators.reverse(globalCI), # global indices
    Iterators.reverse(blockCI),  # block indices
  )
    bi, bj = Tuple(bidx) # block i, j
    gi, gj = Tuple(gidx) .+ solver.nhalo # global i, j
    aᵢ₊½ = solver.mean_func(α[gi, gj], α[gi + 1, gj])
    aᵢ₋½ = solver.mean_func(α[gi, gj], α[gi - 1, gj])
    aⱼ₊½ = solver.mean_func(α[gi, gj], α[gi, gj + 1])
    aⱼ₋½ = solver.mean_func(α[gi, gj], α[gi, gj - 1])

    aᵢⱼ = α[gi, gj]
    aᵢ₊₁ⱼ = α[gi + 1, gj]
    aᵢ₋₁ⱼ = α[gi - 1, gj]
    aᵢⱼ₊₁ = α[gi, gj + 1]
    aᵢⱼ₋₁ = α[gi, gj - 1]

    m = solver.metrics[gi, gj]

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
    qⁿ⁺¹[bi, bj] =
      (
        # orthogonal terms
        (m.ξx^2 + m.ξy^2) *
        (aᵢ₊½ * qⁿ⁺¹[bi + 1, bj] - aᵢ₋½ * (qⁿ[bi, bj] - qⁿ⁺¹[bi - 1, bj])) +
        (m.ηx^2 + m.ηy^2) *
        (aⱼ₊½ * qⁿ⁺¹[bi, bj + 1] - aⱼ₋½ * (qⁿ[bi, bj] - qⁿ⁺¹[bi, bj - 1])) +
        # non-orthogonal terms
        0.25(m.ξx * m.ηx + m.ξy * m.ηy) * (
          aᵢ₊₁ⱼ * (qⁿ[bi + 1, bj + 1] - qⁿ[bi + 1, bj - 1]) -
          aᵢ₋₁ⱼ * (qⁿ[bi - 1, bj + 1] - qⁿ[bi - 1, bj - 1])
        ) +
        0.25(m.ξx * m.ηx + m.ξy * m.ηy) * (
          aᵢⱼ₊₁ * (qⁿ[bi + 1, bj + 1] - qⁿ[bi - 1, bj + 1]) -
          aᵢⱼ₋₁ * (qⁿ[bi + 1, bj - 1] - qⁿ[bi - 1, bj - 1])
        ) +
        # geometric terms
        0.5aᵢⱼ * αᵢⱼ * (qⁿ[bi + 1, bj] - qⁿ⁺¹[bi - 1, bj]) +
        0.5aᵢⱼ * βᵢⱼ * (qⁿ[bi, bj + 1] - qⁿ⁺¹[bi, bj - 1]) +
        # source and remaining terms
        solver.source_term[gi, gj] +
        (qⁿ[bi, bj] / Δt)
      ) / ((1 / Δt) + (aᵢ₊½ * (m.ξx^2 + m.ξy^2) + aⱼ₊½ * (m.ηx^2 + m.ηy^2)))

    qⁿ⁺¹[bi, bj] = cutoff(qⁿ⁺¹[bi, bj])
  end

  # Now average the forward/reverse sweeps
  L₂ = 0.0
  @inline for idx in blockCI
    L₂ += abs2(qⁿ⁺¹[idx] - pⁿ⁺¹[idx])
    u[idx] = 0.5(qⁿ⁺¹[idx] + pⁿ⁺¹[idx])
  end

  for idx in blockCI
    if !isfinite(u[idx]) || u[idx] < 0
      @show u[idx]
      @show qⁿ⁺¹[idx]
      @show pⁿ⁺¹[idx]
      error("Invalid value for u $(u[idx]) at $idx")
    end
  end

  N = length(blockCI)
  L₂ = sqrt(L₂ / N)

  return L₂
end

function nc_solve(solver::BlockADESolver, u, Δt)
  n_blocks = nblocks(u)
  @sync for blockid in 1:n_blocks
    @tspawnat blockid nc_forward_sweep_block!(solver, u, Δt, blockid)
  end

  updatehalo!(solver.pⁿ⁺¹)
  copy_halo!(u, solver.pⁿ⁺¹)

  @sync for blockid in 1:n_blocks
    @tspawnat blockid nc_reverse_sweep_block!(solver, u, Δt, blockid)
  end

  @sync for blockid in 1:n_blocks
    @tspawnat blockid ave_sweeps(solver, u, blockid)
  end

  return nothing
end

function nc_forward_sweep_block!(solver::BlockADESolver, u0, Δt, blockid)
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

  ilo, ihi, jlo, jhi = u0.loop_limits[blockid]
  globalCI = CartesianIndices(u0.global_blockranges[blockid])
  blockCI = CartesianIndices((ilo:ihi, jlo:jhi))
  @assert length(globalCI) == length(blockCI)

  α = solver.a
  # applybc!(u, solver.bcs, solver.nhalo) # update the ghost cell temperatures

  #------------------------------------------------------
  # Forward Sweep
  #------------------------------------------------------
  # anything at i-1 or j-1 is from time level n+1, since we already computed it

  # make alias for code readibilty
  pⁿ⁺¹ = solver.pⁿ⁺¹[blockid]
  u = u0[blockid]
  @inline for idx in eachindex(u)
    pⁿ⁺¹[idx] = u[idx]
  end
  pⁿ = @views pⁿ⁺¹

  for (gidx, bidx) in zip(
    globalCI, # global indices
    blockCI,  # block indices
  )
    # @show bidx, gidx
    bi, bj = Tuple(bidx) # block i, j
    gi, gj = Tuple(gidx) .+ solver.nhalo # global i, j
    aᵢ₊½ = solver.mean_func(α[gi, gj], α[gi + 1, gj])
    aᵢ₋½ = solver.mean_func(α[gi, gj], α[gi - 1, gj])
    aⱼ₊½ = solver.mean_func(α[gi, gj], α[gi, gj + 1])
    aⱼ₋½ = solver.mean_func(α[gi, gj], α[gi, gj - 1])

    aᵢⱼ = α[gi, gj]
    aᵢ₊₁ⱼ = α[gi + 1, gj]
    aᵢ₋₁ⱼ = α[gi - 1, gj]
    aᵢⱼ₊₁ = α[gi, gj + 1]
    aᵢⱼ₋₁ = α[gi, gj - 1]

    # if bi == ihi
    #   @show aᵢ₊₁ⱼ, pⁿ[bi, bj], pⁿ[bi + 1, bj]
    # end
    m = solver.metrics[gi, gj]
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

    pⁿ⁺¹[bi, bj] =
      (
        # orthogonal terms
        (m.ξx^2 + m.ξy^2) *
        (aᵢ₊½ * (pⁿ[bi + 1, bj] - pⁿ[bi, bj]) + aᵢ₋½ * pⁿ⁺¹[bi - 1, bj]) +
        (m.ηx^2 + m.ηy^2) *
        (aⱼ₊½ * (pⁿ[bi, bj + 1] - pⁿ[bi, bj]) + aⱼ₋½ * pⁿ⁺¹[bi, bj - 1]) +
        # non-orthogonal terms
        0.25(m.ξx * m.ηx + m.ξy * m.ηy) * (
          aᵢ₊₁ⱼ * (pⁿ[bi + 1, bj + 1] - pⁿ[bi + 1, bj - 1]) -
          aᵢ₋₁ⱼ * (pⁿ[bi - 1, bj + 1] - pⁿ[bi - 1, bj - 1])
        ) +
        0.25(m.ξx * m.ηx + m.ξy * m.ηy) * (
          aᵢⱼ₊₁ * (pⁿ[bi + 1, bj + 1] - pⁿ[bi - 1, bj + 1]) -
          aᵢⱼ₋₁ * (pⁿ[bi + 1, bj - 1] - pⁿ[bi - 1, bj - 1])
        ) +
        # geometric terms
        0.5aᵢⱼ * αᵢⱼ * (pⁿ[bi + 1, bj] - pⁿ⁺¹[bi - 1, bj]) +
        0.5aᵢⱼ * βᵢⱼ * (pⁿ[bi, bj + 1] - pⁿ⁺¹[bi, bj - 1]) +
        # source and remaining terms
        solver.source_term[gi, gj] +
        (pⁿ[bi, bj] / Δt)
      ) / ((1 / Δt) + (aᵢ₋½ * (m.ξx^2 + m.ξy^2) + aⱼ₋½ * (m.ηx^2 + m.ηy^2)))

    pⁿ⁺¹[bi, bj] = cutoff(pⁿ⁺¹[bi, bj])
  end
end

function nc_reverse_sweep_block!(solver::BlockADESolver, u0, Δt, blockid)
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

  ilo, ihi, jlo, jhi = u0.loop_limits[blockid]
  globalCI = CartesianIndices(u0.global_blockranges[blockid])
  blockCI = CartesianIndices((ilo:ihi, jlo:jhi))
  @assert length(globalCI) == length(blockCI)

  α = solver.a
  # applybc!(u, solver.bcs, solver.nhalo) # update the ghost cell temperatures

  #------------------------------------------------------
  # Forward Sweep
  #------------------------------------------------------
  # anything at i-1 or j-1 is from time level n+1, since we already computed it

  # make alias for code readibilty
  qⁿ⁺¹ = solver.qⁿ⁺¹[blockid]
  u = u0[blockid]
  @inline for idx in eachindex(u)
    qⁿ⁺¹[idx] = u[idx]
  end
  qⁿ = @views qⁿ⁺¹

  #------------------------------------------------------
  # Reverse Sweep
  #------------------------------------------------------
  # anything at i+1 or j+1 is from time level n+1, since we already computed it
  @inline for (gidx, bidx) in zip(
    Iterators.reverse(globalCI), # global indices
    Iterators.reverse(blockCI),  # block indices
  )
    bi, bj = Tuple(bidx) # block i, j
    gi, gj = Tuple(gidx) .+ solver.nhalo # global i, j
    aᵢ₊½ = solver.mean_func(α[gi, gj], α[gi + 1, gj])
    aᵢ₋½ = solver.mean_func(α[gi, gj], α[gi - 1, gj])
    aⱼ₊½ = solver.mean_func(α[gi, gj], α[gi, gj + 1])
    aⱼ₋½ = solver.mean_func(α[gi, gj], α[gi, gj - 1])

    aᵢⱼ = α[gi, gj]
    aᵢ₊₁ⱼ = α[gi + 1, gj]
    aᵢ₋₁ⱼ = α[gi - 1, gj]
    aᵢⱼ₊₁ = α[gi, gj + 1]
    aᵢⱼ₋₁ = α[gi, gj - 1]

    m = solver.metrics[gi, gj]

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
    qⁿ⁺¹[bi, bj] =
      (
        # orthogonal terms
        (m.ξx^2 + m.ξy^2) *
        (aᵢ₊½ * qⁿ⁺¹[bi + 1, bj] - aᵢ₋½ * (qⁿ[bi, bj] - qⁿ⁺¹[bi - 1, bj])) +
        (m.ηx^2 + m.ηy^2) *
        (aⱼ₊½ * qⁿ⁺¹[bi, bj + 1] - aⱼ₋½ * (qⁿ[bi, bj] - qⁿ⁺¹[bi, bj - 1])) +
        # non-orthogonal terms
        0.25(m.ξx * m.ηx + m.ξy * m.ηy) * (
          aᵢ₊₁ⱼ * (qⁿ[bi + 1, bj + 1] - qⁿ[bi + 1, bj - 1]) -
          aᵢ₋₁ⱼ * (qⁿ[bi - 1, bj + 1] - qⁿ[bi - 1, bj - 1])
        ) +
        0.25(m.ξx * m.ηx + m.ξy * m.ηy) * (
          aᵢⱼ₊₁ * (qⁿ[bi + 1, bj + 1] - qⁿ[bi - 1, bj + 1]) -
          aᵢⱼ₋₁ * (qⁿ[bi + 1, bj - 1] - qⁿ[bi - 1, bj - 1])
        ) +
        # geometric terms
        0.5aᵢⱼ * αᵢⱼ * (qⁿ[bi + 1, bj] - qⁿ⁺¹[bi - 1, bj]) +
        0.5aᵢⱼ * βᵢⱼ * (qⁿ[bi, bj + 1] - qⁿ⁺¹[bi, bj - 1]) +
        # source and remaining terms
        solver.source_term[gi, gj] +
        (qⁿ[bi, bj] / Δt)
      ) / ((1 / Δt) + (aᵢ₊½ * (m.ξx^2 + m.ξy^2) + aⱼ₊½ * (m.ηx^2 + m.ηy^2)))

    qⁿ⁺¹[bi, bj] = cutoff(qⁿ⁺¹[bi, bj])
  end

  return nothing
end

function ave_sweeps(solver, u, blockid)
  u_dom = domainview(u, blockid)
  qⁿ⁺¹ = domainview(solver.qⁿ⁺¹, blockid)
  pⁿ⁺¹ = domainview(solver.pⁿ⁺¹, blockid)

  @inline for idx in eachindex(u_dom)
    u_dom[idx] = 0.5(qⁿ⁺¹[idx] + pⁿ⁺¹[idx])
    if !isfinite(u_dom[idx])
      error("invalide u!")
    end
  end

  return nothing
end

# @inline function edge_terms_with_nonorthongal(α_edge, edge_metrics)
#   @unpack αᵢ₊½, αᵢ₋½, αⱼ₊½, αⱼ₋½ = α_edge

#   edge_metric = @views edge_metrics
#   Jᵢ₊½ = edge_metric.Jᵢ₊½
#   Jᵢ₋½ = edge_metric.Jᵢ₋½
#   Jⱼ₊½ = edge_metric.Jⱼ₊½
#   Jⱼ₋½ = edge_metric.Jⱼ₋½
#   Jξx_ᵢ₊½ = edge_metric.Jξx_ᵢ₊½
#   Jξy_ᵢ₊½ = edge_metric.Jξy_ᵢ₊½
#   Jηx_ᵢ₊½ = edge_metric.Jηx_ᵢ₊½
#   Jηy_ᵢ₊½ = edge_metric.Jηy_ᵢ₊½
#   Jξx_ᵢ₋½ = edge_metric.Jξx_ᵢ₋½
#   Jξy_ᵢ₋½ = edge_metric.Jξy_ᵢ₋½
#   Jηx_ᵢ₋½ = edge_metric.Jηx_ᵢ₋½
#   Jηy_ᵢ₋½ = edge_metric.Jηy_ᵢ₋½
#   Jξx_ⱼ₊½ = edge_metric.Jξx_ⱼ₊½
#   Jξy_ⱼ₊½ = edge_metric.Jξy_ⱼ₊½
#   Jηx_ⱼ₊½ = edge_metric.Jηx_ⱼ₊½
#   Jηy_ⱼ₊½ = edge_metric.Jηy_ⱼ₊½
#   Jξx_ⱼ₋½ = edge_metric.Jξx_ⱼ₋½
#   Jξy_ⱼ₋½ = edge_metric.Jξy_ⱼ₋½
#   Jηx_ⱼ₋½ = edge_metric.Jηx_ⱼ₋½
#   Jηy_ⱼ₋½ = edge_metric.Jηy_ⱼ₋½

#   fᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
#   fᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2) / Jᵢ₋½
#   fⱼ₊½ = αⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½
#   fⱼ₋½ = αⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2) / Jⱼ₋½

#   gᵢ₊½ = αᵢ₊½ * (Jξx_ᵢ₊½ * Jηx_ᵢ₊½ + Jξy_ᵢ₊½ * Jηy_ᵢ₊½) / (4Jᵢ₊½)
#   gᵢ₋½ = αᵢ₋½ * (Jξx_ᵢ₋½ * Jηx_ᵢ₋½ + Jξy_ᵢ₋½ * Jηy_ᵢ₋½) / (4Jᵢ₋½)
#   gⱼ₊½ = αⱼ₊½ * (Jξx_ⱼ₊½ * Jηx_ⱼ₊½ + Jξy_ⱼ₊½ * Jηy_ⱼ₊½) / (4Jⱼ₊½)
#   gⱼ₋½ = αⱼ₋½ * (Jξx_ⱼ₋½ * Jηx_ⱼ₋½ + Jξy_ⱼ₋½ * Jηy_ⱼ₋½) / (4Jⱼ₋½)

#   return (; fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½)
# end

# function compute_metrics_fd(mesh, (i, j))
#   xᵢⱼ, yᵢⱼ = coord(mesh, (i + 0.5, j + 0.5))
#   xᵢ₊₁ⱼ, yᵢ₊₁ⱼ = coord(mesh, (i + 1.5, j))
#   xᵢ₊₁ⱼ₊₁, yᵢ₊₁ⱼ₊₁ = coord(mesh, (i + 1.5, j + 1.5))
#   xᵢ₊₁ⱼ₋₁, yᵢ₊₁ⱼ₋₁ = coord(mesh, (i + 1.5, j - 1.5))
#   xᵢ₋₁ⱼ₋₁, yᵢ₋₁ⱼ₋₁ = coord(mesh, (i - 1.5, j - 1.5))
#   xᵢ₋₁ⱼ₊₁, yᵢ₋₁ⱼ₊₁ = coord(mesh, (i - 1.5, j + 1.5))
#   xᵢ₋₁ⱼ, yᵢ₋₁ⱼ = coord(mesh, (i - 1.5, j + 0.5))
#   xᵢⱼ₊₁, yᵢⱼ₊₁ = coord(mesh, (i + 0.5, j + 1.5))
#   xᵢⱼ₋₁, yᵢⱼ₋₁ = coord(mesh, (i + 0.5, j - 1.5))
#   # xᵢⱼ, yᵢⱼ =         coord(mesh, (i, j))
#   # xᵢ₊₁ⱼ, yᵢ₊₁ⱼ =     coord(mesh, (i + 1, j))
#   # xᵢ₊₁ⱼ₊₁, yᵢ₊₁ⱼ₊₁ = coord(mesh, (i + 1, j + 1))
#   # xᵢ₊₁ⱼ₋₁, yᵢ₊₁ⱼ₋₁ = coord(mesh, (i + 1, j - 1))
#   # xᵢ₋₁ⱼ₋₁, yᵢ₋₁ⱼ₋₁ = coord(mesh, (i - 1, j - 1))
#   # xᵢ₋₁ⱼ₊₁, yᵢ₋₁ⱼ₊₁ = coord(mesh, (i - 1, j + 1))
#   # xᵢ₋₁ⱼ, yᵢ₋₁ⱼ =     coord(mesh, (i - 1, j))
#   # xᵢⱼ₊₁, yᵢⱼ₊₁ =     coord(mesh, (i, j + 1))
#   # xᵢⱼ₋₁, yᵢⱼ₋₁ =     coord(mesh, (i, j - 1))

#   J = 0.25 * ((xᵢ₊₁ⱼ - xᵢ₋₁ⱼ) * (yᵢⱼ₊₁ - yᵢⱼ₋₁) - (xᵢⱼ₊₁ - xᵢⱼ₋₁) * (yᵢ₊₁ⱼ - yᵢ₋₁ⱼ))
#   Jᵢ₊½ =
#     0.25 * (
#       (xᵢ₊₁ⱼ - xᵢⱼ) * (yᵢⱼ₊₁ - yᵢⱼ₋₁ + yᵢ₊₁ⱼ₊₁ - yᵢ₊₁ⱼ₋₁) -
#       (yᵢ₊₁ⱼ - yᵢⱼ) * (xᵢⱼ₊₁ - xᵢⱼ₋₁ + xᵢ₊₁ⱼ₊₁ - xᵢ₊₁ⱼ₋₁)
#     )
#   Jᵢ₋½ =
#     0.25 * (
#       (xᵢⱼ - xᵢ₋₁ⱼ) * (yᵢⱼ₊₁ - yᵢⱼ₋₁ + yᵢ₋₁ⱼ₊₁ - yᵢ₋₁ⱼ₋₁) -
#       (yᵢⱼ - yᵢ₋₁ⱼ) * (xᵢⱼ₊₁ - xᵢⱼ₋₁ + xᵢ₋₁ⱼ₊₁ - xᵢ₋₁ⱼ₋₁)
#     )
#   Jⱼ₊½ =
#     0.25 * (
#       (yᵢⱼ₊₁ - yᵢⱼ) * (xᵢ₊₁ⱼ - xᵢ₋₁ⱼ + xᵢ₊₁ⱼ₊₁ - xᵢ₋₁ⱼ₊₁) -
#       (xᵢⱼ₊₁ - xᵢⱼ) * (yᵢ₊₁ⱼ - yᵢ₋₁ⱼ + yᵢ₊₁ⱼ₊₁ - yᵢ₋₁ⱼ₊₁)
#     )
#   Jⱼ₋½ =
#     0.25 * (
#       (yᵢⱼ - yᵢⱼ₋₁) * (xᵢ₊₁ⱼ - xᵢ₋₁ⱼ + xᵢ₊₁ⱼ₋₁ - xᵢ₋₁ⱼ₋₁) -
#       (xᵢⱼ - xᵢⱼ₋₁) * (yᵢ₊₁ⱼ - yᵢ₋₁ⱼ + yᵢ₊₁ⱼ₋₁ - yᵢ₋₁ⱼ₋₁)
#     )

#   ξx = (yᵢⱼ₊₁ - yᵢⱼ₋₁) / (2J)
#   ξy = -(xᵢⱼ₊₁ - xᵢⱼ₋₁) / (2J)
#   ηx = -(yᵢ₊₁ⱼ - yᵢ₋₁ⱼ) / (2J)
#   ηy = (xᵢ₊₁ⱼ - xᵢ₋₁ⱼ) / (2J)

#   ξxᵢ₊½ = (yᵢⱼ₊₁ - yᵢⱼ₋₁ + yᵢ₊₁ⱼ₊₁ - yᵢ₊₁ⱼ₋₁) / (4Jᵢ₊½)
#   ξxᵢ₋½ = (yᵢⱼ₊₁ - yᵢⱼ₋₁ + yᵢ₋₁ⱼ₊₁ - yᵢ₋₁ⱼ₋₁) / (4Jᵢ₋½)
#   ξyᵢ₊½ = (xᵢⱼ₊₁ - xᵢⱼ₋₁ + xᵢ₊₁ⱼ₊₁ - xᵢ₊₁ⱼ₋₁) / (-4Jᵢ₊½)
#   ξyᵢ₋½ = (xᵢⱼ₊₁ - xᵢⱼ₋₁ + xᵢ₋₁ⱼ₊₁ - xᵢ₋₁ⱼ₋₁) / (-4Jᵢ₋½)

#   ηxⱼ₊½ = (yᵢ₊₁ⱼ - yᵢ₋₁ⱼ + yᵢ₊₁ⱼ₊₁ - yᵢ₋₁ⱼ₊₁) / (-4Jⱼ₊½)
#   ηxⱼ₋½ = (yᵢ₊₁ⱼ - yᵢ₋₁ⱼ + yᵢ₊₁ⱼ₋₁ - yᵢ₋₁ⱼ₋₁) / (-4Jⱼ₋½)

#   ηyⱼ₊½ = (xᵢ₊₁ⱼ - xᵢ₋₁ⱼ + xᵢ₊₁ⱼ₊₁ - xᵢ₋₁ⱼ₊₁) / (4Jⱼ₊½)
#   ηyⱼ₋½ = (xᵢ₊₁ⱼ - xᵢ₋₁ⱼ + xᵢ₊₁ⱼ₋₁ - xᵢ₋₁ⱼ₋₁) / (4Jⱼ₋½)

#   ξxⱼ₊½ = (yᵢⱼ₊₁ - yᵢⱼ) / Jⱼ₊½
#   ξxⱼ₋½ = (yᵢⱼ - yᵢⱼ₋₁) / Jⱼ₋½

#   ξyⱼ₊½ = -(xᵢⱼ₊₁ - xᵢⱼ) / Jⱼ₊½
#   ξyⱼ₋½ = -(xᵢⱼ - xᵢⱼ₋₁) / Jⱼ₋½

#   ηxᵢ₊½ = -(yᵢ₊₁ⱼ - yᵢⱼ) / Jᵢ₊½
#   ηxᵢ₋½ = -(yᵢⱼ - yᵢ₋₁ⱼ) / Jᵢ₋½

#   ηyᵢ₊½ = (xᵢ₊₁ⱼ - xᵢⱼ) / Jᵢ₊½
#   ηyᵢ₋½ = (xᵢⱼ - xᵢ₋₁ⱼ) / Jᵢ₋½

#   cell_metrics = (;
#     # J,
#     ξx,
#     ξy,
#     ηx,
#     ηy,
#     ξxᵢ₋½,
#     ξyᵢ₋½,
#     ηxᵢ₋½,
#     ηyᵢ₋½,
#     ξxⱼ₋½,
#     ξyⱼ₋½,
#     ηxⱼ₋½,
#     ηyⱼ₋½,
#     ξxᵢ₊½,
#     ξyᵢ₊½,
#     ηxᵢ₊½,
#     ηyᵢ₊½,
#     ξxⱼ₊½,
#     ξyⱼ₊½,
#     ηxⱼ₊½,
#     ηyⱼ₊½,
#   )

#   return cell_metrics
# end
