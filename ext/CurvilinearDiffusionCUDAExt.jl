module CurvilinearDiffusionCUDAExt

using CurvilinearDiffusion
using CurvilinearGrids
using KernelAbstractions

using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSR

export initialize_coefficient_matrix

function CurvilinearDiffusion.initialize_coefficient_matrix(
  mesh::CurvilinearGrid2D, ::CUDABackend
)
  return CuSparseMatrixCSR(init_A_matrix(mesh, CPU()))
end

end
