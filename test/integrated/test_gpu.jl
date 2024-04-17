using SparseArrays, Krylov, LinearOperators, KrylovPreconditioners
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using LinearAlgebra

# CPU Arrays
ni, nj = 2000, 2000
len = ni * nj

A_cpu = spdiagm(
  -ni - 1 => zeros(len - ni - 1), # (i-1, j-1)
  -ni => zeros(len - ni),     # (i  , j-1)
  -ni + 1 => zeros(len - ni + 1), # (i+1, j-1)
  -1 => -ones(len - 1),      # (i-1, j  )
  0 => rand(len),           # (i  , j  )
  1 => ones(len - 1),      # (i+1, j  )
  ni - 1 => zeros(len - ni + 1), # (i-1, j+1)
  ni => zeros(len - ni),     # (i  , j+1)
  ni + 1 => zeros(len - ni - 1), # (i+1, j+1)
)

b_cpu = rand(len)

# Transfer the linear system from the CPU to the GPU
A_gpu = CuSparseMatrixCSR(A_cpu)  # A_gpu = CuSparseMatrixCSC(A_cpu)
b_gpu = CuVector(b_cpu)

# ILU(0) decomposition LU ≈ A for CuSparseMatrixCSC or CuSparseMatrixCSR matrices
P = ilu02(A_gpu)

# Additional vector required for solving triangular systems
n = length(b_gpu)
T = eltype(b_gpu)
z = CUDA.zeros(T, n)

P = kp_ilu0(A_gpu);

x_gpu, stats = gmres(A_gpu, b_gpu; N=P, ldiv=true)
