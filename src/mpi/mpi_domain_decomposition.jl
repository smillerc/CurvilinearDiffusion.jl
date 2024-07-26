module MPIDomainDecomposition

using MPI
using CartesianDomains
using NVTX

include("mpi_topology.jl")
using .ParallelTopologyTypes
export CartesianTopology

export neighbor, on_boundary
export ilo_neighbor, ihi_neighbor, jlo_neighbor, jhi_neighbor, klo_neighbor, khi_neighbor
export updatehalo!

include("halo_exchange.jl")

end