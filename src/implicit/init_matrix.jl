
function _initialize_coefficent_matrix(dims::NTuple{2,Int}, bcs)
  m, n = dims

  nhalo = 1
  ninner = (m - 2nhalo) * (n - 2nhalo)

  # 2 coeffs per boundary loc
  ilo = (m - 2nhalo) * (bcs.ilo isa NeumannBC)
  ihi = (m - 2nhalo) * (bcs.ihi isa NeumannBC)
  jlo = (n - 2nhalo) * (bcs.jlo isa NeumannBC)
  jhi = (n - 2nhalo) * (bcs.jhi isa NeumannBC)

  # 8 coeffs per inner loc (not including the main diagonal)
  inner = ninner * 8
  diag = m * n

  nzvals = (ihi + ilo + jlo + jhi + diag + inner)

  rows = zeros(Int, nzvals)
  cols = zeros(Int, nzvals)
  vals = zeros(nzvals)

  k = 0
  CI = CartesianIndices((m, n))
  LI = LinearIndices(CI)

  # main-diagonal
  for idx in CI
    k += 1
    row = LI[idx]
    rows[k] = row
    cols[k] = row
    vals[k] = 1
  end

  innerCI = expand(CI, -1)
  for idx in innerCI
    i, j = idx.I
    row = LI[idx]
    for joff in (-1, 0, 1)
      for ioff in (-1, 0, 1)
        ijk = (i + ioff, j + joff)
        col = LI[ijk...]
        if row != col # already did the diagonals previously
          k += 1
          rows[k] = row
          cols[k] = col
          vals[k] = 0
        end
      end
    end
  end

  if ilo > 0
    ilo_CI = @view CI[begin, (begin + 1):(end - 1)]
    for idx in ilo_CI
      k += 1
      i, j = idx.I
      row = LI[idx]
      col = LI[i + 1, j]

      rows[k] = row
      cols[k] = col
      vals[k] = -1
    end
  end

  if ihi > 0
    ihi_CI = @view CI[end, (begin + 1):(end - 1)]
    for idx in ihi_CI
      k += 1
      i, j = idx.I
      row = LI[idx]
      col = LI[i - 1, j]

      rows[k] = row
      cols[k] = col
      vals[k] = -1
    end
  end

  if jlo > 0
    jlo_CI = @view CI[(begin + 1):(end - 1), begin]
    for idx in jlo_CI
      k += 1
      i, j = idx.I
      row = LI[idx]
      col = LI[i, j + 1]

      rows[k] = row
      cols[k] = col
      vals[k] = -1
    end
  end
  if jhi > 0
    jhi_CI = @view CI[(begin + 1):(end - 1), end]
    for idx in jhi_CI
      k += 1
      i, j = idx.I
      row = LI[idx]
      col = LI[i, j - 1]

      rows[k] = row
      cols[k] = col
      vals[k] = -1
    end
  end

  A = sparse(rows, cols, vals)

  return A
end

function _initialize_coefficent_matrix(dims::NTuple{3,Int}, bcs)
  ni, nj, nk = dims
  nhalo = 1
  ninner = (ni - 2nhalo) * (nj - 2nhalo) * (nk - 2nhalo)

  # 2 coeffs per boundary loc
  ilo = ((nj - 2nhalo) * (nk - 2nhalo)) * (bcs.ilo isa NeumannBC)
  ihi = ((nj - 2nhalo) * (nk - 2nhalo)) * (bcs.ihi isa NeumannBC)
  jlo = ((ni - 2nhalo) * (nk - 2nhalo)) * (bcs.jlo isa NeumannBC)
  jhi = ((ni - 2nhalo) * (nk - 2nhalo)) * (bcs.jhi isa NeumannBC)
  klo = ((ni - 2nhalo) * (nj - 2nhalo)) * (bcs.klo isa NeumannBC)
  khi = ((ni - 2nhalo) * (nj - 2nhalo)) * (bcs.khi isa NeumannBC)

  # 26 coeffs per inner loc (not including the main diagonal)
  inner = ninner * 26
  diag = ni * nj * nk

  nzvals = (ihi + ilo + jlo + jhi + klo + khi + diag + inner)

  rows = zeros(Int, nzvals)
  cols = zeros(Int, nzvals)
  vals = zeros(nzvals)

  z = 0
  CI = CartesianIndices((ni, nj, nk))
  LI = LinearIndices(CI)

  # main-diagonal
  for idx in CI
    z += 1
    row = LI[idx]
    rows[z] = cols[z] = row
    vals[z] = 1
  end

  innerCI = expand(CI, -1)
  for idx in innerCI
    i, j, k = idx.I
    row = LI[idx]
    for koff in (-1, 0, 1)
      for joff in (-1, 0, 1)
        for ioff in (-1, 0, 1)
          ijk = (i + ioff, j + joff, k + koff)
          col = LI[ijk...]
          if row != col # already did the diagonals previously
            z += 1
            rows[z] = row
            cols[z] = col
            vals[z] = 0
          end
        end
      end
    end
  end

  if ilo > 0
    ilo_CI = @view CI[begin, (begin + 1):(end - 1), (begin + 1):(end - 1)]
    for idx in ilo_CI
      z += 1
      i, j, k = idx.I
      row = LI[idx]
      col = LI[i + 1, j, k]

      rows[z] = row
      cols[z] = col
      vals[z] = -1
    end
  end

  if ihi > 0
    ihi_CI = @view CI[end, (begin + 1):(end - 1), (begin + 1):(end - 1)]
    for idx in ihi_CI
      z += 1
      i, j, k = idx.I
      row = LI[idx]
      col = LI[i - 1, j, k]

      rows[z] = row
      cols[z] = col
      vals[z] = -1
    end
  end

  if jlo > 0
    jlo_CI = @view CI[(begin + 1):(end - 1), begin, (begin + 1):(end - 1)]
    for idx in jlo_CI
      z += 1
      i, j, k = idx.I
      row = LI[idx]
      col = LI[i, j + 1, k]

      rows[z] = row
      cols[z] = col
      vals[z] = -1
    end
  end

  if jhi > 0
    jhi_CI = @view CI[(begin + 1):(end - 1), end, (begin + 1):(end - 1)]
    for idx in jhi_CI
      z += 1
      i, j, k = idx.I
      row = LI[idx]
      col = LI[i, j - 1, k]

      rows[z] = row
      cols[z] = col
      vals[z] = -1
    end
  end

  if klo > 0
    klo_CI = @view CI[(begin + 1):(end - 1), (begin + 1):(end - 1), begin]
    for idx in klo_CI
      z += 1
      i, j, k = idx.I
      row = LI[idx]
      col = LI[i, j, k + 1]

      rows[z] = row
      cols[z] = col
      vals[z] = -1
    end
  end

  if khi > 0
    khi_CI = @view CI[(begin + 1):(end - 1), (begin + 1):(end - 1), end]
    for idx in khi_CI
      z += 1
      i, j, k = idx.I
      row = LI[idx]
      col = LI[i, j, k - 1]

      rows[z] = row
      cols[z] = col
      vals[z] = -1
    end
  end

  A = sparse(rows, cols, vals)
  #   A = sparsecsr(rows, cols, vals)

  return A
end

# function initialize_coefficient_matrix(iterators, ::CurvilinearGrid1D, ::CPU)
#   len = length(iterators.full.linear)
#   A = Tridiagonal(zeros(len - 1), ones(len), zeros(len - 1))
#   return A, nothing
# end

function initialize_coefficient_matrix(iterators, ::CurvilinearGrid2D, bcs, ::CPU)
  dims = size(iterators.full.cartesian)
  A = _initialize_coefficent_matrix(dims, bcs)
  return A
end

function initialize_coefficient_matrix(iterators, ::CurvilinearGrid3D, bcs, ::CPU)
  dims = size(iterators.full.cartesian)
  A = _initialize_coefficent_matrix(dims, bcs)
  return A
end

# function init_coeff_matrix(dims::NTuple{N,Int}; format=:csc, cardinal_only=false) where {N}
#   @assert all(dims .> 2)

#   @assert format == :csr || format == :csc

#   CI = CartesianIndices(dims)
#   LI = LinearIndices(CI)

#   if cardinal_only
#     stencil_col = get_cardinal_stencil_cols(LI)
#   else
#     stencil_col = get_full_stencil_cols(LI)
#   end

#   kv = ntuple(
#     i -> stencil_col[i][1] => ones(stencil_col[i][2]) * Int(stencil_col[i][1] == 0),
#     length(stencil_col),
#   )

#   if format === :csr
#     I, J, V, mmax, nmax = SparseArrays.spdiagm_internal(kv...)
#     A = sparsecsr(I, J, V, mmax, nmax)
#   else
#     A = spdiagm(kv...)
#   end

#   return A, stencil_col
# end

# # helper function to determin diagonal offsets and lengths in the sparse matrices
# function diag_offset(LI::LinearIndices, offset::NTuple{2,Int})
#   ijk = (2, 2)
#   idx = @. ijk + offset
#   diagonal_len = length(LI)
#   diagonal_offset = -(LI[ijk...] - LI[idx...])

#   vector_len = diagonal_len - abs(diagonal_offset)
#   if diagonal_offset == 0
#     data = vector_len
#   else
#     data = vector_len
#   end

#   return diagonal_offset => data
# end

# function diag_offset(LI::LinearIndices, offset::NTuple{3,Int})
#   ijk = (2, 2, 2)
#   idx = @. ijk + offset
#   diagonal_len = length(LI)
#   diagonal_offset = -(LI[ijk...] - LI[idx...])

#   vector_len = diagonal_len - abs(diagonal_offset)
#   if diagonal_offset == 0
#     data = ones(vector_len)
#   else
#     data = rand(vector_len)
#   end

#   return diagonal_offset => vector_len
# end

# function get_cardinal_stencil_cols(LI::LinearIndices{2})
#   stencil_cols = (
#     ᵢⱼ₋₁=diag_offset(LI, (+0, -1)),
#     ᵢ₋₁ⱼ=diag_offset(LI, (-1, +0)),
#     ᵢⱼ=diag_offset(LI, (+0, +0)),
#     ᵢ₊₁ⱼ=diag_offset(LI, (+1, +0)),
#     ᵢⱼ₊₁=diag_offset(LI, (+0, +1)),
#   )

#   return stencil_cols
# end

# function get_cardinal_stencil_cols(LI::LinearIndices{3})
#   stencil_cols = (
#     ᵢⱼₖ₋₁=diag_offset(LI, (+0, +0, -1)),
#     ᵢⱼ₋₁ₖ=diag_offset(LI, (+0, -1, +0)),
#     ᵢ₋₁ⱼₖ=diag_offset(LI, (-1, +0, +0)),
#     ᵢⱼₖ=diag_offset(LI, (+0, +0 + 0)),
#     ᵢ₊₁ⱼₖ=diag_offset(LI, (+1, +0, +0)),
#     ᵢⱼ₊₁ₖ=diag_offset(LI, (+0, +1, +0)),
#     ᵢⱼₖ₊₁=diag_offset(LI, (+0, +0, +1)),
#   )

#   return stencil_cols
# end

# function get_full_stencil_cols(LI::LinearIndices{2})
#   stencil_cols = (
#     ᵢ₋₁ⱼ₋₁=diag_offset(LI, (-1, -1)),
#     ᵢⱼ₋₁=diag_offset(LI, (+0, -1)),
#     ᵢ₊₁ⱼ₋₁=diag_offset(LI, (+1, -1)),
#     ᵢ₋₁ⱼ=diag_offset(LI, (-1, +0)),
#     ᵢⱼ=diag_offset(LI, (+0, +0)),
#     ᵢ₊₁ⱼ=diag_offset(LI, (+1, +0)),
#     ᵢ₋₁ⱼ₊₁=diag_offset(LI, (-1, +1)),
#     ᵢⱼ₊₁=diag_offset(LI, (+0, +1)),
#     ᵢ₊₁ⱼ₊₁=diag_offset(LI, (+1, +1)),
#   )
#   return stencil_cols
# end

# function get_full_stencil_cols(LI::LinearIndices{3})
#   stencil_cols = (
#     ᵢ₋₁ⱼ₋₁ₖ₋₁=diag_offset(LI, (-1, -1, -1)),
#     ᵢⱼ₋₁ₖ₋₁=diag_offset(LI, (+0, -1, -1)),
#     ᵢ₊₁ⱼ₋₁ₖ₋₁=diag_offset(LI, (+1, -1, -1)),
#     ᵢ₋₁ⱼₖ₋₁=diag_offset(LI, (-1, +0, -1)),
#     ᵢⱼₖ₋₁=diag_offset(LI, (+0, +0, -1)),
#     ᵢ₊₁ⱼₖ₋₁=diag_offset(LI, (+1, +0, -1)),
#     ᵢ₋₁ⱼ₊₁ₖ₋₁=diag_offset(LI, (-1, +1, -1)),
#     ᵢⱼ₊₁ₖ₋₁=diag_offset(LI, (+0, +1, -1)),
#     ᵢ₊₁ⱼ₊₁ₖ₋₁=diag_offset(LI, (+1, +1, -1)),
#     ᵢ₋₁ⱼ₋₁ₖ=diag_offset(LI, (-1, -1, +0)),
#     ᵢⱼ₋₁ₖ=diag_offset(LI, (+0, -1, +0)),
#     ᵢ₊₁ⱼ₋₁ₖ=diag_offset(LI, (+1, -1, +0)),
#     ᵢ₋₁ⱼₖ=diag_offset(LI, (-1, +0, +0)),
#     ᵢⱼₖ=diag_offset(LI, (+0, +0, +0)),
#     ᵢ₊₁ⱼₖ=diag_offset(LI, (+1, +0, +0)),
#     ᵢ₋₁ⱼ₊₁ₖ=diag_offset(LI, (-1, +1, +0)),
#     ᵢⱼ₊₁ₖ=diag_offset(LI, (+0, +1, +0)),
#     ᵢ₊₁ⱼ₊₁ₖ=diag_offset(LI, (+1, +1, +0)),
#     ᵢ₋₁ⱼ₋₁ₖ₊₁=diag_offset(LI, (-1, -1, +1)),
#     ᵢⱼ₋₁ₖ₊₁=diag_offset(LI, (+0, -1, +1)),
#     ᵢ₊₁ⱼ₋₁ₖ₊₁=diag_offset(LI, (+1, -1, +1)),
#     ᵢ₋₁ⱼₖ₊₁=diag_offset(LI, (-1, +0, +1)),
#     ᵢⱼₖ₊₁=diag_offset(LI, (+0, +0, +1)),
#     ᵢ₊₁ⱼₖ₊₁=diag_offset(LI, (+1, +0, +1)),
#     ᵢ₋₁ⱼ₊₁ₖ₊₁=diag_offset(LI, (-1, +1, +1)),
#     ᵢⱼ₊₁ₖ₊₁=diag_offset(LI, (+0, +1, +1)),
#     ᵢ₊₁ⱼ₊₁ₖ₊₁=diag_offset(LI, (+1, +1, +1)),
#   )
#   return stencil_cols
# end
