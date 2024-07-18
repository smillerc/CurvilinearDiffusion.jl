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

function CurvilinearDiffusion.assemble!(
  A::CuSparseMatrixCSR{Tv,Ti},
  u::AbstractArray{T,N},
  scheme::ImplicitScheme{N,T,BE},
  mesh,
  Δt::Real,
) where {Tv,Ti,N,T,BE}

  #
  m, _ = size(A)

  kernel = @cuda launch = false assemble_csr!(
    A,
    scheme.linear_problem.b,
    scheme.source_term,
    u,
    scheme.α,
    Δt,
    mesh.cell_center_metrics.J,
    mesh.edge_metrics,
    scheme.iterators.mesh,
    scheme.iterators.full.cartesian,
    scheme.limits,
    scheme.mean_func,
    scheme.bcs,
  )

  config = launch_configuration(kernel.fun)
  threads = min(m, config.threads)

  blocks = cld(m, threads)

  kernel(
    A,
    scheme.linear_problem.b,
    scheme.source_term,
    u,
    scheme.α,
    Δt,
    mesh.cell_center_metrics.J,
    mesh.edge_metrics,
    scheme.iterators.mesh,
    scheme.iterators.full.cartesian,
    scheme.limits,
    scheme.mean_func,
    scheme.bcs;
    threads,
    blocks,
  )
  #   CUDA.synchronize()
  return nothing
end

function assemble_csr!(
  A,
  b::AbstractVector{T},
  source_term::AbstractArray{T,2},
  u::AbstractArray{T,2},
  α::AbstractArray{T,2},
  Δt,
  cell_center_jacobian,
  edge_metrics,
  mesh_indices,
  diffusion_prob_indices,
  limits,
  mean_func::F,
  bcs,
) where {T,F<:Function}

  # every thread processes an entire row
  row = threadIdx().x + (blockIdx().x - 1) * blockDim().x

  nrows = size(A, 1)

  row > nrows && return nothing

  @unpack ilo, ihi, jlo, jhi = limits

  begin
    mesh_idx = mesh_indices[row]
    diff_idx = diffusion_prob_indices[row]
    i, j = diff_idx.I

    onbc = i == ilo || i == ihi || j == jlo || j == jhi

    if !onbc
      J = cell_center_jacobian[mesh_idx]
      edge_α = edge_diffusivity(α, diff_idx, mean_func)
      A_coeffs = _inner_diffusion_operator(edge_α, Δt, J, edge_metrics, mesh_idx)
      rhs_coeff = (source_term[diff_idx] * Δt + u[mesh_idx]) * J

      cfirst = A.rowPtr[row]
      @inbounds for c in 1:9
        zidx = cfirst + c - 1
        A.nzVal[zidx] = A_coeffs[c]
      end

    else
      rhs_coeff = bc_operator(bcs, diff_idx, limits, T)
    end

    b[row] = rhs_coeff
  end

  return nothing
end

function assemble_csr!(
  A,
  b::AbstractVector{T},
  source_term::AbstractArray{T,3},
  u::AbstractArray{T,3},
  α::AbstractArray{T,3},
  Δt,
  cell_center_jacobian,
  edge_metrics,
  mesh_indices,
  diffusion_prob_indices,
  limits,
  mean_func::F,
  bcs,
) where {T,F<:Function}

  # every thread processes an entire row
  row = threadIdx().x + (blockIdx().x - 1) * blockDim().x

  nrows = size(A, 1)

  row > nrows && return nothing

  @unpack ilo, ihi, jlo, jhi, klo, khi = limits

  begin
    mesh_idx = mesh_indices[row]
    diff_idx = diffusion_prob_indices[row]
    i, j, k = diff_idx.I

    onbc = i == ilo || i == ihi || j == jlo || j == jhi || k == klo || k == khi

    if !onbc
      J = cell_center_jacobian[mesh_idx]
      edge_α = edge_diffusivity(α, diff_idx, mean_func)
      A_coeffs = _inner_diffusion_operator(edge_α, Δt, J, edge_metrics, mesh_idx)
      rhs_coeff = (source_term[diff_idx] * Δt + u[mesh_idx]) * J

      cfirst = A.rowPtr[row]
      @inbounds for c in 1:27
        zidx = cfirst + c - 1
        A.nzVal[zidx] = A_coeffs[c]
      end

    else
      rhs_coeff = CurvilinearDiffusion.ImplicitSchemeType.bc_operator(
        bcs, diff_idx, limits, T
      )
    end

    b[row] = rhs_coeff
  end

  return nothing
end
