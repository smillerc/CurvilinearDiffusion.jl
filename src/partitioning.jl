module Partitioning

using Base.Threads
using LinearAlgebra

export tile_indices_1d, tile_indices_2d, tile_indices_3d

"""Return all common denominators of n"""
function denominators(n::Integer)
  denominators = Vector{Int}(undef, 0)
  for i in 1:n
    if mod(n, i) == 0
      push!(denominators, i)
    end
  end
  return denominators
end

"""Returns the optimal number of tiles in (i,j) given total number of tiles n"""
function num_2d_tiles(n)
  # find all common denominators of the total number of images
  denoms = denominators(n)

  # find all combinations of common denominators
  # whose product equals the total number of images
  dim1 = Vector{Int}(undef, 0)
  dim2 = Vector{Int}(undef, 0)
  for j in eachindex(denoms)
    for i in eachindex(denoms)
      if denoms[i] * denoms[j] == n
        push!(dim1, denoms[i])
        push!(dim2, denoms[j])
      end
    end
  end
  # pick the set of common denominators with the minimal norm
  # between two elements -- rectangle closest to a square
  num_2d_tiles = [dim1[1], dim2[1]]
  for i in 2:length(dim1)
    n1 = norm([dim1[i], dim2[i]] .- sqrt(n))
    n2 = norm(num_2d_tiles .- sqrt(n))
    if n1 < n2
      num_2d_tiles = [dim1[i], dim2[i]]
    end
  end
  return num_2d_tiles
end

"""Returns the optimal number of tiles in (i,j,k) given total number of tiles n"""
function num_3d_tiles(n)
  # find all common denominators of the total number of images
  denoms = denominators(n)

  # find all combinations of common denominators
  # whose product equals the total number of images
  dim1 = Vector{Int}(undef, 0)
  dim2 = Vector{Int}(undef, 0)
  dim3 = Vector{Int}(undef, 0)
  for k in eachindex(denoms)
    for j in eachindex(denoms)
      for i in eachindex(denoms)
        if denoms[i] * denoms[j] * denoms[k] == n
          push!(dim1, denoms[i])
          push!(dim2, denoms[j])
          push!(dim3, denoms[k])
        end
      end
    end
  end
  # pick the set of common denominators with the minimal norm
  # between two elements -- rectangle closest to a square
  num_3d_tiles = [dim1[1], dim2[1], dim3[1]]
  for i in 2:length(dim1)
    n1 = norm([dim1[i], dim2[i], dim3[i]] .- sqrt(n))
    n2 = norm(num_3d_tiles .- sqrt(n))
    if n1 < n2
      num_3d_tiles = [dim1[i], dim2[i], dim3[i]]
    end
  end
  return num_3d_tiles
end

"""
Given an input I dimensions of the total computational domain,
returns an array of start and end indices [ilo,ihi]
"""
function tile_indices_1d(dims::Integer; ntiles=nthreads(), id=threadid())
  tile_size = dims ÷ ntiles

  # start and end indices assuming equal tile sizes
  ilo = (id - 1) * tile_size + 1
  ihi = ilo + tile_size - 1

  # if we have any remainder, distribute it to the tiles at the end
  offset = ntiles - mod(dims, ntiles)
  if id > offset
    ilo = ilo + id - offset - 1
    ihi = ihi + id - offset
  end
  return (ilo, ihi)
end

"""
Given an input (I,J) dimensions of the total computational domain,
returns an array of start and end indices [ilo,ihi,jlo,jhi]
"""
function tile_indices_2d(dims; ntiles=nthreads(), id=threadid())
  tiles = num_2d_tiles(ntiles)
  tiles_ij = tile_id_to_ij(id; ntiles=ntiles)
  ilo, ihi = tile_indices_1d(dims[1]; id=tiles_ij[1], ntiles=tiles[1])
  jlo, jhi = tile_indices_1d(dims[2]; id=tiles_ij[2], ntiles=tiles[2])
  return (ilo, ihi, jlo, jhi)
end

"""
Given an input (I,J,K) dimensions of the total computational domain,
returns an array of start and end indices [ilo,ihi,jlo,jhi,klo,khi]
"""
function tile_indices_3d(dims; ntiles=nthreads(), id=threadid())
  indices = zeros(Int, 6)
  tiles = num_3d_tiles(ntiles)
  tiles_ij = tile_id_to_ijk(id; ntiles=ntiles)
  ilo, ihi = tile_indices_1d(dims[1]; id=tiles_ij[1], ntiles=tiles[1])
  jlo, jhi = tile_indices_1d(dims[2]; id=tiles_ij[2], ntiles=tiles[2])
  klo, khi = tile_indices_1d(dims[3]; id=tiles_ij[3], ntiles=tiles[3])
  return (ilo, ihi, jlo, jhi, klo, khi)
end

"""Given tile id in a 1D layout, returns the corresponding tile indices in a 2D layout"""
function tile_id_to_ij(id; ntiles=nthreads())
  if id < 1
    @error("Invalid tile id")
  end
  I, J = num_2d_tiles(ntiles)
  CI = CartesianIndices((1:I, 1:J))
  ij = Tuple(CI[id])
  return ij
end

"""Given tile id in a 1D layout, returns the corresponding tile indices in a 3D layout"""
function tile_id_to_ijk(id; ntiles=nthreads())
  if id < 1
    @error("Invalid tile id")
  end
  I, J, K = num_3d_tiles(ntiles)
  CI = CartesianIndices((1:I, 1:J, 1:K))
  ijk = Tuple(CI[id])
  return ijk
end

function partition_array!(A::AbstractArray{T,2}) where {T<:Number}
  ilo, ihi, jlo, jhi = tile_indices_2d(size(A))
  subarr = @view A[ilo:ihi, jlo:jhi]
  subarr .= threadid()
  return nothing
end

function partition_array!(A::AbstractArray{T,3}) where {T<:Number}
  ilo, ihi, jlo, jhi, klo, khi = tile_indices_3d(size(A))
  subarr = @view A[ilo:ihi, jlo:jhi, klo:khi]
  subarr .= threadid()
  return nothing
end

end # module
