module CurvilinearDiffusionCUDAExt

using CurvilinearDiffusion
using CurvilinearGrids
using KernelAbstractions
using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSR

include("CUDA/assembly.jl")

function CurvilinearDiffusion.initialize_coefficient_matrix(
  iterators, mesh, bcs, ::CUDABackend
)
  return CuSparseMatrixCSR(
    CurvilinearDiffusion.initialize_coefficient_matrix(
      initialize_coefficient_matrix(iterators, mesh, bcs, CPU())
    ),
  )
end

end
