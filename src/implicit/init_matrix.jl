
function initialize_coefficient_matrix(mesh::CurvilinearGrid1D, ::CPU)
  len, = cellsize(mesh)
  A = Tridiagonal(zeros(len - 1), ones(len), zeros(len - 1))
  return A, nothing
end

function initialize_coefficient_matrix(mesh::CurvilinearGrid2D, ::CPU)
  dims = cellsize(mesh)
  A, stencil_col_lookup = init_coeff_matrix(dims; format=:csc, cardinal_only=false)
  return A, stencil_col_lookup
end

# function initialize_coefficient_matrix(mesh::CurvilinearGrid2D, ::CUDABackend)
#   A, stencil_col_lookup = initialize_coefficient_matrix(mesh, CPU())

#   return CuSparseMatrixCSR(A), stencil_col_lookup
# end

function initialize_coefficient_matrix(mesh::CurvilinearGrid3D, ::CPU)
  dims = cellsize(mesh)
  A, stencil_col_lookup = init_coeff_matrix(dims; format=:csc, cardinal_only=false)
  return A, stencil_col_lookup
end

function init_coeff_matrix(dims::NTuple{N,Int}; format=:csr, cardinal_only=false) where {N}
  @assert all(dims .> 2)

  CI = CartesianIndices(dims)
  LI = LinearIndices(CI)

  if cardinal_only
    stencil_col = get_cardinal_stencil_cols(LI)
  else
    stencil_col = get_full_stencil_cols(LI)
  end

  kv = ntuple(
    i -> stencil_col[i][1] => ones(stencil_col[i][2]) * Int(stencil_col[i][1] == 0),
    length(stencil_col),
  )

  if format === :csr
    I, J, V, mmax, nmax = SparseArrays.spdiagm_internal(kv...)
    A = sparsecsr(I, J, V, mmax, nmax)
  else
    A = spdiagm(kv...)
  end

  return A, stencil_col
end

# helper function to determin diagonal offsets and lengths in the sparse matrices
function diag_offset(LI::LinearIndices, offset::NTuple{2,Int})
  ijk = (2, 2)
  idx = @. ijk + offset
  diagonal_len = length(LI)
  diagonal_offset = -(LI[ijk...] - LI[idx...])

  vector_len = diagonal_len - abs(diagonal_offset)
  if diagonal_offset == 0
    data = vector_len
  else
    data = vector_len
  end

  return diagonal_offset => data
end

function diag_offset(LI::LinearIndices, offset::NTuple{3,Int})
  ijk = (2, 2, 2)
  idx = @. ijk + offset
  diagonal_len = length(LI)
  diagonal_offset = -(LI[ijk...] - LI[idx...])

  vector_len = diagonal_len - abs(diagonal_offset)
  if diagonal_offset == 0
    data = ones(vector_len)
  else
    data = rand(vector_len)
  end

  return diagonal_offset => vector_len
end

function get_cardinal_stencil_cols(LI::LinearIndices{2})
  stencil_cols = (
    ᵢⱼ₋₁=diag_offset(LI, (+0, -1)),
    ᵢ₋₁ⱼ=diag_offset(LI, (-1, +0)),
    ᵢⱼ=diag_offset(LI, (+0, +0)),
    ᵢ₊₁ⱼ=diag_offset(LI, (+1, +0)),
    ᵢⱼ₊₁=diag_offset(LI, (+0, +1)),
  )

  return stencil_cols
end

function get_cardinal_stencil_cols(LI::LinearIndices{3})
  stencil_cols = (
    ᵢⱼₖ₋₁=diag_offset(LI, (+0, +0, -1)),
    ᵢⱼ₋₁ₖ=diag_offset(LI, (+0, -1, +0)),
    ᵢ₋₁ⱼₖ=diag_offset(LI, (-1, +0, +0)),
    ᵢⱼₖ=diag_offset(LI, (+0, +0 + 0)),
    ᵢ₊₁ⱼₖ=diag_offset(LI, (+1, +0, +0)),
    ᵢⱼ₊₁ₖ=diag_offset(LI, (+0, +1, +0)),
    ᵢⱼₖ₊₁=diag_offset(LI, (+0, +0, +1)),
  )

  return stencil_cols
end

function get_full_stencil_cols(LI::LinearIndices{2})
  stencil_cols = (
    ᵢ₋₁ⱼ₋₁=diag_offset(LI, (-1, -1)),
    ᵢⱼ₋₁=diag_offset(LI, (+0, -1)),
    ᵢ₊₁ⱼ₋₁=diag_offset(LI, (+1, -1)),
    ᵢ₋₁ⱼ=diag_offset(LI, (-1, +0)),
    ᵢⱼ=diag_offset(LI, (+0, +0)),
    ᵢ₊₁ⱼ=diag_offset(LI, (+1, +0)),
    ᵢ₋₁ⱼ₊₁=diag_offset(LI, (-1, +1)),
    ᵢⱼ₊₁=diag_offset(LI, (+0, +1)),
    ᵢ₊₁ⱼ₊₁=diag_offset(LI, (+1, +1)),
  )
  return stencil_cols
end

function get_full_stencil_cols(LI::LinearIndices{3})
  stencil_cols = (
    ᵢ₋₁ⱼ₋₁ₖ₋₁=diag_offset(LI, (-1, -1, -1)),
    ᵢⱼ₋₁ₖ₋₁=diag_offset(LI, (+0, -1, -1)),
    ᵢ₊₁ⱼ₋₁ₖ₋₁=diag_offset(LI, (+1, -1, -1)),
    ᵢ₋₁ⱼₖ₋₁=diag_offset(LI, (-1, +0, -1)),
    ᵢⱼₖ₋₁=diag_offset(LI, (+0, +0, -1)),
    ᵢ₊₁ⱼₖ₋₁=diag_offset(LI, (+1, +0, -1)),
    ᵢ₋₁ⱼ₊₁ₖ₋₁=diag_offset(LI, (-1, +1, -1)),
    ᵢⱼ₊₁ₖ₋₁=diag_offset(LI, (+0, +1, -1)),
    ᵢ₊₁ⱼ₊₁ₖ₋₁=diag_offset(LI, (+1, +1, -1)),
    ᵢ₋₁ⱼ₋₁ₖ=diag_offset(LI, (-1, -1, +0)),
    ᵢⱼ₋₁ₖ=diag_offset(LI, (+0, -1, +0)),
    ᵢ₊₁ⱼ₋₁ₖ=diag_offset(LI, (+1, -1, +0)),
    ᵢ₋₁ⱼₖ=diag_offset(LI, (-1, +0, +0)),
    ᵢⱼₖ=diag_offset(LI, (+0, +0, +0)),
    ᵢ₊₁ⱼₖ=diag_offset(LI, (+1, +0, +0)),
    ᵢ₋₁ⱼ₊₁ₖ=diag_offset(LI, (-1, +1, +0)),
    ᵢⱼ₊₁ₖ=diag_offset(LI, (+0, +1, +0)),
    ᵢ₊₁ⱼ₊₁ₖ=diag_offset(LI, (+1, +1, +0)),
    ᵢ₋₁ⱼ₋₁ₖ₊₁=diag_offset(LI, (-1, -1, +1)),
    ᵢⱼ₋₁ₖ₊₁=diag_offset(LI, (+0, -1, +1)),
    ᵢ₊₁ⱼ₋₁ₖ₊₁=diag_offset(LI, (+1, -1, +1)),
    ᵢ₋₁ⱼₖ₊₁=diag_offset(LI, (-1, +0, +1)),
    ᵢⱼₖ₊₁=diag_offset(LI, (+0, +0, +1)),
    ᵢ₊₁ⱼₖ₊₁=diag_offset(LI, (+1, +0, +1)),
    ᵢ₋₁ⱼ₊₁ₖ₊₁=diag_offset(LI, (-1, +1, +1)),
    ᵢⱼ₊₁ₖ₊₁=diag_offset(LI, (+0, +1, +1)),
    ᵢ₊₁ⱼ₊₁ₖ₊₁=diag_offset(LI, (+1, +1, +1)),
  )
  return stencil_cols
end
