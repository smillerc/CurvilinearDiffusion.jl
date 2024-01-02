using CurvilinearGrids
using CurvilinearDiffusion
using BlockHaloArrays
using .Threads

nt = nthreads()

function uniform_grid(nx, ny)
  x0, x1 = (0, 1)
  y0, y1 = (0, 1)

  x(i, j) = @. x0 + (x1 - x0) * ((i - 1) / (nx - 1))
  y(i, j) = @. y0 + (y1 - y0) * ((j - 1) / (ny - 1))

  return (x, y)
end

ni, nj = (101, 101)
nhalo = 1
x, y = uniform_grid(ni, nj)
mesh = CurvilinearGrid2D(x, y, (ni, nj), nhalo)

bcs = (ilo=:zero_flux, ihi=:zero_flux, jlo=:zero_flux, jhi=:zero_flux)

solver = BlockADESolver(mesh, bcs)

dims = (8, 3) # a 30 x 20 matrix
nhalo = 1 # number of halo entries in each dimension
n_blocks = 2 # nthreads() # nblocks must be ≤ nthreads() or a warning will be issued

A = BlockHaloArray(dims, nhalo, n_blocks; T=Float64)

ilo, ihi, jlo, jhi = A.loop_limits[2]
blockCI = CartesianIndices((ilo:ihi, jlo:jhi))
globalCI = CartesianIndices(A.global_blockranges[2])

globalCI[1].I .+ nhalo
# B = BlockHaloArray(dims, nhalo, nblocks; T=Float64)

function haloviews(A, bid)
  return (
    haloview(A, bid, :ilo),
    haloview(A, bid, :jlo),
    haloview(A, bid, :jhi),
    haloview(A, bid, :ihi),
    haloview(A, bid, :ilojlo),
    haloview(A, bid, :ilojhi),
    haloview(A, bid, :ihijlo),
    haloview(A, bid, :ihijhi),
  )
end

# A1 = domainview(A, 1)
# A1 .= 1
# A[1]
# B[1]
# copy_domain!(B, A)

# bid = 2
# GI = CartesianIndices(A.global_blockranges[bid])

# ilo, ihi, jlo, jhi = A.loop_limits[bid]

# BI = CartesianIndices((ilo:ihi, jlo:jhi))

# length(GI)
# length(BI)

# for (gidx, bidx) in zip(GI, BI)
#   @show gidx, bidx
# end
