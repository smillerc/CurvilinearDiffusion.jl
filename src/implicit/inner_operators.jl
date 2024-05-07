# ---------------------------------------------------------------------------
#  
# ---------------------------------------------------------------------------
# @kernel function inner_diffusion_op_kernel_2d!(
#   A,
#   α,
#   Δt,
#   cell_center_metrics,
#   edge_metrics,
#   grid_indices,
#   matrix_indices,
#   meanfunc::F,
#   stencil_col_lookup,
# ) where {F}
#   idx = @index(Global, Linear)

const offsets1d = (-1, 0, 1)

const offsets2d = (
  (-1, -1), (+0, -1), (+1, -1), (-1, +0), (+0, +0), (+1, +0), (-1, +1), (+0, +1), (+1, +1)
)

const offsets3d = (
  (-1, -1, -1),
  (+0, -1, -1),
  (+1, -1, -1),
  (-1, +0, -1),
  (+0, +0, -1),
  (+1, +0, -1),
  (-1, +1, -1),
  (+0, +1, -1),
  (+1, +1, -1),
  (-1, -1, +0),
  (+0, -1, +0),
  (+1, -1, +0),
  (-1, +0, +0),
  (+0, +0, +0),
  (+1, +0, +0),
  (-1, +1, +0),
  (+0, +1, +0),
  (+1, +1, +0),
  (-1, -1, +1),
  (+0, -1, +1),
  (+1, -1, +1),
  (-1, +0, +1),
  (+0, +0, +1),
  (+1, +0, +1),
  (-1, +1, +1),
  (+0, +1, +1),
  (+1, +1, +1),
)

# ---------------------------------------------------------------------------
#  
# ---------------------------------------------------------------------------

@kernel function _assembly_kernel_2d(
  A::SparseMatrixCSC{T,Ti},
  b::AbstractVector{T},
  source_term::AbstractArray{T,N},
  u::AbstractArray{T,N},
  α::AbstractArray{T,N},
  Δt,
  cell_center_jacobian,
  edge_metrics,
  mesh_indices,
  diffusion_prob_indices,
  matrix_indices,
  limits,
  mean_func::F,
  bcs,
) where {T,Ti,N,F<:Function}

  # These are the indicies corresponding to the edge
  # of the diffusion problem
  @unpack ilo, ihi, jlo, jhi = limits

  idx = @index(Global, Linear)

  @inbounds begin
    row = matrix_indices[idx]
    mesh_idx = mesh_indices[idx]
    diff_idx = diffusion_prob_indices[idx]
    i, j = diff_idx.I

    onbc = i == ilo || i == ihi || j == jlo || j == jhi

    if !onbc
      J = cell_center_jacobian[mesh_idx]
      edge_α = edge_diffusivity(α, diff_idx, mean_func)
      A_coeffs = _inner_diffusion_operator(edge_α, Δt, J, edge_metrics, mesh_idx)

      rhs_coeff = (source_term[diff_idx] * Δt + u[mesh_idx]) * J

      @inbounds for icol in 1:9
        ij = diff_idx.I .+ offsets2d[icol]
        col = matrix_indices[ij...]

        A[row, col] = A_coeffs[icol]
      end

    else
      rhs_coeff = bc_operator(bcs, diff_idx, limits, T)
    end

    b[row] = rhs_coeff
  end
end

@kernel function _assembly_kernel_3d(
  A::SparseMatrixCSC{T,Ti},
  b::AbstractVector{T},
  source_term::AbstractArray{T,N},
  u::AbstractArray{T,N},
  α::AbstractArray{T,N},
  Δt,
  cell_center_jacobian,
  edge_metrics,
  mesh_indices,
  diffusion_prob_indices,
  matrix_indices,
  limits,
  mean_func::F,
  bcs,
) where {T,Ti,N,F<:Function}

  # These are the indicies corresponding to the edge
  # of the diffusion problem
  @unpack ilo, ihi, jlo, jhi, klo, khi = limits

  idx = @index(Global, Linear)

  @inbounds begin
    row = matrix_indices[idx]
    mesh_idx = mesh_indices[idx]
    diff_idx = diffusion_prob_indices[idx]
    i, j, k = diff_idx.I

    onbc = i == ilo || i == ihi || j == jlo || j == jhi || k == klo || k == khi

    if !onbc
      J = cell_center_jacobian[mesh_idx]
      edge_α = edge_diffusivity(α, diff_idx, mean_func)
      A_coeffs = _inner_diffusion_operator(edge_α, Δt, J, edge_metrics, mesh_idx)

      rhs_coeff = (source_term[diff_idx] * Δt + u[mesh_idx]) * J

      @inbounds for icol in 1:27
        ijk = diff_idx.I .+ offsets3d[icol]
        col = matrix_indices[ijk...]

        A[row, col] = A_coeffs[icol]
      end

    else
      rhs_coeff = bc_operator(bcs, diff_idx, limits, T)
    end

    b[row] = rhs_coeff
  end
end

# ---------------------------------------------------------------------------
#  Boundary condition RHS operators
# ---------------------------------------------------------------------------

function bc_operator(bcs, idx::CartesianIndex{1}, limits, T)
  @unpack ilo, ihi = limits
  i, = idx.I

  at_ibc = i == ilo || i == ihi
  at_bc = at_ibc || at_jbc

  if !at_bc
    error("The bc_operator is getting called, but we're not at_ a boundary!")
  end

  if i == ilo
    rhs_coeff = bc_rhs_coefficient(bcs.ilo, idx, T)
  elseif i == ihi
    rhs_coeff = bc_rhs_coefficient(bcs.ihi, idx, T)
  end

  return rhs_coeff
end

function bc_operator(bcs, idx::CartesianIndex{2}, limits, T)
  @unpack ilo, ihi, jlo, jhi = limits
  i, j = idx.I

  at_ibc = i == ilo || i == ihi
  at_jbc = j == jlo || j == jhi
  at_bc = at_ibc || at_jbc

  if !at_bc
    error("The bc_operator is getting called, but we're not at_ a boundary!")
  end

  if i == ilo
    rhs_coeff = bc_rhs_coefficient(bcs.ilo, idx, T)
  elseif i == ihi
    rhs_coeff = bc_rhs_coefficient(bcs.ihi, idx, T)
  elseif j == jlo
    rhs_coeff = bc_rhs_coefficient(bcs.jlo, idx, T)
  elseif j == jhi
    rhs_coeff = bc_rhs_coefficient(bcs.jhi, idx, T)
  end

  return rhs_coeff
end

function bc_operator(bcs, idx::CartesianIndex{3}, limits, T)
  @unpack ilo, ihi, jlo, jhi, klo, khi = limits
  i, j, k = idx.I

  at_ibc = i == ilo || i == ihi
  at_jbc = j == jlo || j == jhi
  at_kbc = k == klo || k == khi
  at_bc = at_ibc || at_jbc || at_kbc

  if !at_bc
    error("The bc_operator is getting called, but we're not at_ a boundary!")
  end

  if i == ilo
    rhs_coeff = bc_rhs_coefficient(bcs.ilo, idx, T)
  elseif i == ihi
    rhs_coeff = bc_rhs_coefficient(bcs.ihi, idx, T)
  elseif j == jlo
    rhs_coeff = bc_rhs_coefficient(bcs.jlo, idx, T)
  elseif j == jhi
    rhs_coeff = bc_rhs_coefficient(bcs.jhi, idx, T)
  elseif k == klo
    rhs_coeff = bc_rhs_coefficient(bcs.klo, idx, T)
  elseif k == khi
    rhs_coeff = bc_rhs_coefficient(bcs.khi, idx, T)
  end

  return rhs_coeff
end

# ---------------------------------------------------------------------------
#  Edge diffusivity
# ---------------------------------------------------------------------------

function edge_diffusivity(α, idx::CartesianIndex{1}, mean_function::F) where {F<:Function}
  i, = idx.I
  edge_diffusivity = (
    αᵢ₊½=mean_function(α[i], α[i + 1]), #
    αᵢ₋½=mean_function(α[i], α[i - 1]), #
  )

  return edge_diffusivity
end

function edge_diffusivity(α, idx::CartesianIndex{2}, mean_function::F) where {F<:Function}
  i, j = idx.I
  edge_diffusivity = (
    αᵢ₊½=mean_function(α[i, j], α[i + 1, j]),
    αᵢ₋½=mean_function(α[i, j], α[i - 1, j]),
    αⱼ₊½=mean_function(α[i, j], α[i, j + 1]),
    αⱼ₋½=mean_function(α[i, j], α[i, j - 1]),
  )

  return edge_diffusivity
end

function edge_diffusivity(α, idx::CartesianIndex{3}, mean_function::F) where {F<:Function}
  i, j, k = idx.I
  edge_diffusivity = (
    αᵢ₊½=mean_function(α[i, j, k], α[i + 1, j, k]),
    αᵢ₋½=mean_function(α[i, j, k], α[i - 1, j, k]),
    αⱼ₊½=mean_function(α[i, j, k], α[i, j + 1, k]),
    αⱼ₋½=mean_function(α[i, j, k], α[i, j - 1, k]),
    αₖ₊½=mean_function(α[i, j, k], α[i, j, k + 1]),
    αₖ₋½=mean_function(α[i, j, k], α[i, j, k - 1]),
  )

  return edge_diffusivity
end

# ---------------------------------------------------------------------------
#  Inner-domain Operators
# ---------------------------------------------------------------------------

# Generate a stencil for a single 1D cell in the interior
@inline function _inner_diffusion_operator(
  edge_diffusivity, Δt, J, edge_metrics, idx::CartesianIndex{1}
)
  edge_terms = conservative_edge_terms(edge_diffusivity, edge_metrics, idx)
  stencil = stencil_1d(edge_terms, J, Δt)
  return stencil
end

# Generate a stencil for a single 2D cell in the interior
@inline function _inner_diffusion_operator(
  edge_diffusivity, Δt, J, edge_metrics, idx::CartesianIndex{2}
)
  edge_terms = conservative_edge_terms(edge_diffusivity, edge_metrics, idx)
  stencil = stencil_2d(edge_terms, J, Δt)
  return stencil
end

# Generate a stencil for a single 3D cell in the interior
@inline function _inner_diffusion_operator(
  edge_diffusivity, Δt, J, edge_metrics, idx::CartesianIndex{3}
)
  edge_terms = conservative_edge_terms(edge_diffusivity, edge_metrics, idx)
  stencil = stencil_3d(edge_terms, J, Δt)
  return stencil
end
