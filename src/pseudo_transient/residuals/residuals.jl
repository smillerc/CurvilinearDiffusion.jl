
include("orthogonal_mesh_gpu.jl")
# include("non_orthogonal_mesh.jl")

function L2_norm(A::AbstractArray)
  _norm = sqrt(mapreduce(x -> (x^2), +, A) / length(A))
  return _norm
end

function L2_norm(A::Array)
  _L2_norm(A, Val(nthreads()))
end

function _L2_norm(a, ::Val{nchunks}) where {nchunks}
  _numer = @MVector zeros(nchunks)

  @batch for idx in eachindex(a)
    ichunk = threadid()

    _numer[ichunk] += a[idx]^2
  end

  return sqrt(sum(_numer) / length(a))
end

# @kernel function _update_resid!(
#   residuals,
#   cell_center_metrics,
#   edge_metrics,
#   @Const(u),
#   @Const(u_prev),
#   @Const(flux),
#   @Const(source_term),
#   @Const(dt),
#   @Const(I0),
# )
#   idx = @index(Global, Cartesian)
#   idx += I0

#   @inbounds begin
#     @inline ∇q = flux_divergence(flux, cell_center_metrics, edge_metrics, idx)

#     residuals[idx] = -(u[idx] - u_prev[idx]) / dt - ∇q + source_term[idx]
#   end
# end

function update_residual!(
  solver::PseudoTransientSolver{N,T,BE}, mesh, Δt
) where {N,T,BE<:GPU}
  if mesh.is_orthogonal
    update_residuals_orthogonal_2d(solver, mesh, Δt)
  else
    error("hag!")
    # update_residuals_orthogonal_2d(solver, mesh, Δt)
  end
end

# function update_residual!(
#   solver::PseudoTransientSolver{N,T,BE}, mesh::CurvilinearGrid2D, Δt
# ) where {N,T,BE<:CPU}

#   #
#   domain = solver.iterators.domain.cartesian
#   u = solver.u
#   u_prev = solver.u_prev
#   qᵢ, qⱼ = solver.q′
#   cell_center_metrics = mesh.cell_center_metrics
#   edge_metrics = mesh.edge_metrics
#   residuals = solver.res
#   source_term = solver.source_term

#   ξ_x = cell_center_metrics.ξ.x₁ #[i, j]
#   ξ_y = cell_center_metrics.ξ.x₂ #[i, j]
#   η_x = cell_center_metrics.η.x₁ #[i, j]
#   η_y = cell_center_metrics.η.x₂ #[i, j]

#   @batch for idx in domain
#     i, j = idx.I

#     # Jᵢ₊½ = inv(edge_metrics.i₊½.J[i, j])
#     # Jⱼ₊½ = inv(edge_metrics.j₊½.J[i, j])
#     # Jᵢ₋½ = inv(edge_metrics.i₊½.J[i - 1, j])
#     # Jⱼ₋½ = inv(edge_metrics.j₊½.J[i, j - 1])

#     ξx = ξ_x[i, j] #cell_center_metrics.ξ.x₁[i, j]
#     ξy = ξ_y[i, j] #cell_center_metrics.ξ.x₂[i, j]
#     ηx = η_x[i, j] #cell_center_metrics.η.x₁[i, j]
#     ηy = η_y[i, j] #cell_center_metrics.η.x₂[i, j]

#     # ξxᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[i, j] * Jᵢ₊½
#     # ξyᵢ₊½ = edge_metrics.i₊½.ξ̂.x₂[i, j] * Jᵢ₊½
#     # ηxᵢ₊½ = edge_metrics.i₊½.η̂.x₁[i, j] * Jᵢ₊½
#     # ηyᵢ₊½ = edge_metrics.i₊½.η̂.x₂[i, j] * Jᵢ₊½

#     # ξxᵢ₋½ = edge_metrics.i₊½.ξ̂.x₁[i - 1, j] * Jᵢ₋½
#     # ξyᵢ₋½ = edge_metrics.i₊½.ξ̂.x₂[i - 1, j] * Jᵢ₋½
#     # ηxᵢ₋½ = edge_metrics.i₊½.η̂.x₁[i - 1, j] * Jᵢ₋½
#     # ηyᵢ₋½ = edge_metrics.i₊½.η̂.x₂[i - 1, j] * Jᵢ₋½

#     # ξxⱼ₊½ = edge_metrics.j₊½.ξ̂.x₁[i, j] * Jⱼ₊½
#     # ξyⱼ₊½ = edge_metrics.j₊½.ξ̂.x₂[i, j] * Jⱼ₊½
#     # ηxⱼ₊½ = edge_metrics.j₊½.η̂.x₁[i, j] * Jⱼ₊½
#     # ηyⱼ₊½ = edge_metrics.j₊½.η̂.x₂[i, j] * Jⱼ₊½

#     # ξxⱼ₋½ = edge_metrics.j₊½.ξ̂.x₁[i, j - 1] * Jⱼ₋½
#     # ξyⱼ₋½ = edge_metrics.j₊½.ξ̂.x₂[i, j - 1] * Jⱼ₋½
#     # ηxⱼ₋½ = edge_metrics.j₊½.η̂.x₁[i, j - 1] * Jⱼ₋½
#     # ηyⱼ₋½ = edge_metrics.j₊½.η̂.x₂[i, j - 1] * Jⱼ₋½

#     # # flux divergence

#     # aᵢⱼ = (
#     #   ξx * (ξxᵢ₊½ - ξxᵢ₋½) +
#     #   ξy * (ξyᵢ₊½ - ξyᵢ₋½) +
#     #   ηx * (ξxⱼ₊½ - ξxⱼ₋½) +
#     #   ηy * (ξyⱼ₊½ - ξyⱼ₋½)
#     # )

#     # bᵢⱼ = (
#     #   ξx * (ηxᵢ₊½ - ηxᵢ₋½) +
#     #   ξy * (ηyᵢ₊½ - ηyᵢ₋½) +
#     #   ηx * (ηxⱼ₊½ - ηxⱼ₋½) +
#     #   ηy * (ηyⱼ₊½ - ηyⱼ₋½)
#     # )

#     ∂qᵢ∂ξ = (ξx^2 + ξy^2) * (qᵢ[i, j] - qᵢ[i - 1, j])
#     ∂qⱼ∂η = (ηx^2 + ηy^2) * (qⱼ[i, j] - qⱼ[i, j - 1])

#     # ∂qᵢ∂η =
#     #   0.25(ηx * ξx + ηy * ξy) * (
#     #     (qᵢ[i, j + 1] + qᵢ[i - 1, j + 1]) - # take average on either side
#     #     (qᵢ[i, j - 1] + qᵢ[i - 1, j - 1])   # and do diff in j
#     #   )

#     # ∂qⱼ∂ξ =
#     #   0.25(ηx * ξx + ηy * ξy) * (
#     #     (qⱼ[i + 1, j] + qⱼ[i + 1, j - 1]) - # take average on either side
#     #     (qⱼ[i - 1, j] + qⱼ[i - 1, j - 1])   # and do diff in i
#     #   )

#     # ∂H∂ξ = aᵢⱼ * 0.5(qᵢ[i, j] + qᵢ[i - 1, j]) # ∂u/∂ξ + non-orth terms
#     # ∂H∂η = bᵢⱼ * 0.5(qⱼ[i, j] + qⱼ[i, j - 1]) # ∂u/∂η + non-orth terms

#     ∇q = ∂qᵢ∂ξ + ∂qⱼ∂η #+ ∂qᵢ∂η + ∂qⱼ∂ξ + ∂H∂ξ + ∂H∂η

#     residuals[idx] = -(u[idx] - u_prev[idx]) / Δt - ∇q + source_term[idx]
#   end

#   return nothing
# end