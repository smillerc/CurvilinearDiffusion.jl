
include("orthogonal_mesh_resids_cpu.jl")
include("orthogonal_mesh_resids_gpu.jl")

include("nonorthogonal_mesh_resids_gpu.jl")
include("nonorthogonal_mesh_resids_cpu.jl")

function L2_norm(A::AbstractArray)
  _norm = norm(A) / sqrt(length(A))
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

function update_residual!(solver::PseudoTransientSolver{N,T,BE}, mesh, Δt) where {N,T,BE}
  if mesh.is_orthogonal
    update_residuals_orthogonal!(solver, mesh, Δt)
  else
    update_residuals_nonorthogonal!(solver, mesh, Δt)
  end
end
