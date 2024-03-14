module CurvilinearDiffusionExt

using CurvilinearDiffusion
using CurvilinearGrids
using KernelAbstractions

using CUDA.CUSPARSE: CuSparseMatrixCSR

function init_A_matrix(mesh::CurvilinearGrid2D, ::GPU)
  return CuSparseMatrixCSR(init_A_matrix(mesh, CPU()))
end

end
