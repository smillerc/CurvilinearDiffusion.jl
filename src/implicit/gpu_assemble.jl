
# function csr_reduce_kernel(
#   f::F, op::OP, neutral, output::CuDeviceArray, args...
# ) where {F,OP}

#   # every thread processes an entire row; only need
#   # to use 1D indexing, since we're accessing entire matrix rows
#   row = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
#   row > size(output, 1) && return nothing

#   iter = @inbounds CSRIterator{Int}(row, args...)

#   val = op(neutral, neutral)

#   # reduce the values for this row
#   for (col, ptrs) in iter
#     I = CartesianIndex(row, col)
#     vals = ntuple(Val(length(args))) do i
#       arg = @inbounds args[i]
#       ptr = @inbounds ptrs[i]
#       _getindex(arg, I, ptr)
#     end
#     val = op(val, f(vals...))
#   end

#   @inbounds output[row] = val
#   return nothing
# end

# function assemble(A) # A::CUDA.CUSPARSE.CuSparseMatrixCSR

#   # every thread processes an entire row; only need
#   # to use 1D indexing, since we're accessing entire matrix rows
#   row = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
#   row > size(output, 1) && return nothing

#   iter = @inbounds CSRIterator{Int}(row, A)

#   # reduce the values for this row
#   for (col, ptrs) in iter
#     I = CartesianIndex(row, col)

#     vals = ntuple(Val(length(args))) do i
#       arg = @inbounds args[i]
#       ptr = @inbounds ptrs[i]
#       _getindex(arg, I, ptr)
#     end
#   end

#   return nothing
# end

using SparseArrays, SparseMatricesCSR

begin
  ni, nj = 5, 10
  len = ni * nj

  # CI = CartesianIndices((nj, ni))
  # inner_domain = expand(CI, -1)
  # LI = LinearIndices(CI)
  # inner_LI = LI[inner_domain]

  #! format: off
  kv = (
    -ni - 1 => zeros(len - ni - 1), # (i-1, j-1)
    -ni     => zeros(len - ni),     # (i  , j-1)
    -ni + 1 => zeros(len - ni + 1), # (i+1, j-1)
    -1      => -2ones(len - 1),     # (i-1, j  )
    0       => ones(len),           # (i  , j  )
    1       => 2ones(len - 1),      # (i+1, j  )
    ni - 1  => zeros(len - ni + 1), # (i-1, j+1)
    ni      => zeros(len - ni),     # (i  , j+1)
    ni + 1  => zeros(len - ni - 1), # (i+1, j+1)
  )
  #! format: on
# spdiagm(kv...)
  I, J, V, mmax, nmax = SparseArrays.spdiagm_internal(kv...)

  B = sparsecsr(I, J, V, mmax, nmax)
  A = sparse(I, J, V, mmax, nmax)
  nothing
end
