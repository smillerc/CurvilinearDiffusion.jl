using SparseArrays, SparseMatricesCSR

# begin
#   ni, nj, nk = (100, 100, 100)

#   arr = zeros(ni, nj, nk)
#   CI = CartesianIndices(arr)
#   LI = LinearIndices(CI)
#   len = ni * nj * nk
# end

# function init_coeff_matrix((ni, nj, nk)::NTuple{3,Int}; format=:csr, cardinal_only=false)
#   @assert ni > 2 && nj > 2 && nk > 2

#   CI = CartesianIndices((ni, nj, nk))
#   LI = LinearIndices(CI)

#   if cardinal_only
#     kv = (
#       diag_offset(LI, (+0, +0, -1)),  # uⁿ⁺¹ᵢⱼₖ₋₁
#       diag_offset(LI, (+0, -1, +0)),  # uⁿ⁺¹ᵢⱼ₋₁ₖ
#       diag_offset(LI, (-1, +0, +0)),  # uⁿ⁺¹ᵢ₋₁ⱼₖ
#       diag_offset(LI, (+0, +0, +0)),  # uⁿ⁺¹ᵢⱼₖ
#       diag_offset(LI, (+1, +0, +0)),  # uⁿ⁺¹ᵢ₊₁ⱼₖ
#       diag_offset(LI, (+0, +1, +0)),  # uⁿ⁺¹ᵢⱼ₊₁ₖ
#       diag_offset(LI, (+0, +0, +1)),  # uⁿ⁺¹ᵢⱼₖ₊₁
#     )
#   else
#     kv = (
#       diag_offset(LI, (-1, -1, -1)),  # uⁿ⁺¹ᵢ₋₁ⱼ₋₁ₖ₋₁
#       diag_offset(LI, (+0, -1, -1)),  # uⁿ⁺¹ᵢⱼ₋₁ₖ₋₁
#       diag_offset(LI, (+1, -1, -1)),  # uⁿ⁺¹ᵢ₊₁ⱼ₋₁ₖ₋₁
#       diag_offset(LI, (-1, +0, -1)),  # uⁿ⁺¹ᵢ₋₁ⱼₖ₋₁
#       diag_offset(LI, (+0, +0, -1)),  # uⁿ⁺¹ᵢⱼₖ₋₁
#       diag_offset(LI, (+1, +0, -1)),  # uⁿ⁺¹ᵢ₊₁ⱼₖ₋₁
#       diag_offset(LI, (-1, +1, -1)),  # uⁿ⁺¹ᵢ₋₁ⱼ₊₁ₖ₋₁
#       diag_offset(LI, (+0, +1, -1)),  # uⁿ⁺¹ᵢⱼ₊₁ₖ₋₁
#       diag_offset(LI, (+1, +1, -1)),  # uⁿ⁺¹ᵢ₊₁ⱼ₊₁ₖ₋₁
#       diag_offset(LI, (-1, -1, +0)),  # uⁿ⁺¹ᵢ₋₁ⱼ₋₁ₖ
#       diag_offset(LI, (+0, -1, +0)),  # uⁿ⁺¹ᵢⱼ₋₁ₖ
#       diag_offset(LI, (+1, -1, +0)),  # uⁿ⁺¹ᵢ₊₁ⱼ₋₁ₖ
#       diag_offset(LI, (-1, +0, +0)),  # uⁿ⁺¹ᵢ₋₁ⱼₖ
#       diag_offset(LI, (+0, +0, +0)),  # uⁿ⁺¹ᵢⱼₖ
#       diag_offset(LI, (+1, +0, +0)),  # uⁿ⁺¹ᵢ₊₁ⱼₖ
#       diag_offset(LI, (-1, +1, +0)),  # uⁿ⁺¹ᵢ₋₁ⱼ₊₁ₖ
#       diag_offset(LI, (+0, +1, +0)),  # uⁿ⁺¹ᵢⱼ₊₁ₖ
#       diag_offset(LI, (+1, +1, +0)),  # uⁿ⁺¹ᵢ₊₁ⱼ₊₁ₖ
#       diag_offset(LI, (-1, -1, +1)),  # uⁿ⁺¹ᵢ₋₁ⱼ₋₁ₖ₊₁
#       diag_offset(LI, (+0, -1, +1)),  # uⁿ⁺¹ᵢⱼ₋₁ₖ₊₁
#       diag_offset(LI, (+1, -1, +1)),  # uⁿ⁺¹ᵢ₊₁ⱼ₋₁ₖ₊₁
#       diag_offset(LI, (-1, +0, +1)),  # uⁿ⁺¹ᵢ₋₁ⱼₖ₊₁
#       diag_offset(LI, (+0, +0, +1)),  # uⁿ⁺¹ᵢⱼₖ₊₁
#       diag_offset(LI, (+1, +0, +1)),  # uⁿ⁺¹ᵢ₊₁ⱼₖ₊₁
#       diag_offset(LI, (-1, +1, +1)),  # uⁿ⁺¹ᵢ₋₁ⱼ₊₁ₖ₊₁
#       diag_offset(LI, (+0, +1, +1)),  # uⁿ⁺¹ᵢⱼ₊₁ₖ₊₁
#       diag_offset(LI, (+1, +1, +1)),  # uⁿ⁺¹ᵢ₊₁ⱼ₊₁ₖ₊₁
#     )
#   end

#   if format === :csr
#     I, J, V, mmax, nmax = SparseArrays.spdiagm_internal(kv...)
#     A = sparsecsr(I, J, V, mmax, nmax)
#   else
#     A = spdiagm(kv...)
#   end

#   return A
# end

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

# helper function to determin diagonal offsets and lengths
# in the sparse CSR matrix  
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

# begin
#   ni, nj = (15, 5)
#   len = ni * nj
#   A = init_coeff_matrix((ni, nj); format=:csc, cardinal_only=true)
# end

begin
  dims = (30, 3, 3)
  len = prod(dims)
  A, sc = init_coeff_matrix(dims; format=:csc, cardinal_only=false)
  A
end
