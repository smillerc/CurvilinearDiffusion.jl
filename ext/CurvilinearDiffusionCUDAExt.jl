module CurvilinearDiffusionCUDAExt

using CurvilinearDiffusion
using CurvilinearDiffusion.ImplicitSchemeType:
  bc_operator, _inner_diffusion_operator, edge_diffusivity

using CurvilinearGrids
using KernelAbstractions
using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSR
using UnPack

include("CUDA/assembly.jl")

function CurvilinearDiffusion.ImplicitSchemeType.initialize_coefficient_matrix(
  iterators, mesh, bcs, ::CUDABackend
)
  return CuSparseMatrixCSR(
    CurvilinearDiffusion.ImplicitSchemeType.initialize_coefficient_matrix(
      iterators, mesh, bcs, CPU()
    ),
  )
end

end
