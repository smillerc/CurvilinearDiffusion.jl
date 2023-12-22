using CurvilinearGrids
using CurvilinearDiffusion
using .Threads

nt = nthreads()

A = zeros(50, 50);
CurvilinearDiffusion.Partitioning.partition_array!(A)
A

dims = size(A)

ranges = Vector{CartesianIndices}(undef, nt)
for tid in 1:nt
  ilo, ihi, jlo, jhi = CurvilinearDiffusion.Partitioning.tile_indices_2d(
    dims; ntiles=nt, id=tid
  )
  CI = CartesianIndices((ilo:ihi, jlo:jhi))
  ranges[tid] = CI
  @show tid, CI
end

ranges
