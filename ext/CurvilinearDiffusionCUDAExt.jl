module CurvilinearDiffusionCUDAExt

using CurvilinearDiffusion
using CurvilinearGrids
using KernelAbstractions
using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSR

function CurvilinearDiffusion.initialize_coefficient_matrix(
  mesh::CurvilinearGrid2D, ::CUDABackend
)
  return CuSparseMatrixCSR(CurvilinearDiffusion.initialize_coefficient_matrix(mesh, CPU()))
end

function CurvilinearDiffusion.initialize_coefficient_matrix(
  mesh::CurvilinearGrid3D, ::CUDABackend
)
  return CuSparseMatrixCSR(CurvilinearDiffusion.initialize_coefficient_matrix(mesh, CPU()))
end

function CurvilinearDiffusion.assemble_matrix!(
  A::CuSparseMatrixCSR, scheme::ImplicitScheme{2}, mesh, Δt
)
  return nothing
end

function CurvilinearDiffusion.assemble_matrix!(
  A::CuSparseMatrixCSR, scheme::ImplicitScheme{3}, mesh, Δt
)
  return nothing
end

end
