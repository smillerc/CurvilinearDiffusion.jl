module PartitionedImplicitSchemeType

using CurvilinearGrids: onbc, AbstractCurvilinearGrid
using MPIHaloArrays: updatehalo!
using MPI

struct PartitionedImplicitScheme{IS}
  local_scheme::IS
  rank::Int
end

function PartitionedImplicitScheme(
  mesh::AbstractCurvilinearGrid,
  bcs,
  rank::Int;
  direct_solve=false,
  face_conductivity::Symbol=:harmonic,
  T=Float64,
  backend=CPU(),
)

  # if the local boundary condition (bc) is between blocks, then impose a dirichlet, 
  # otherwise use the global bc
  is_const = false # allow the inner dirichlet BCs to have non-constant values
  local_bcs = (
    ilo=onbc(mesh, :ilo) ? bcs.ilo : DirichletBC(-1.0, is_const),
    ihi=onbc(mesh, :ihi) ? bcs.ihi : DirichletBC(-1.0, is_const),
    jlo=onbc(mesh, :jlo) ? bcs.jlo : DirichletBC(-1.0, is_const),
    jhi=onbc(mesh, :jhi) ? bcs.jhi : DirichletBC(-1.0, is_const),
  )

  local_scheme = ImplicitScheme(
    mesh,
    local_bcs;
    direct_solve=direct_solve,
    face_conductivity=face_conductivity,
    T=T,
    backend,
  )

  return PartitionedImplicitScheme(local_scheme, rank)
end

function solve!(scheme::PartitionedImplicitScheme, mesh, u, Δt, comm; kwargs...)
  it = 0
  global_err = Inf
  next_Δt = -Inf
  tol = 1e-6

  it_min = 2 # must do at least 2 iterations

  # outer loop
  while it_min < it < it_max && global_err < tol

    # exchange edge terms
    updatehalo!(u)

    # inner solve
    stats, local_next_Δt = ImplicitSchemeType.solve!(
      scheme.local_scheme, mesh, u, Δt; kwargs...
    )

    # local convergence err
    L₂norm = last(stats.residuals)

    # no need for barrier for inner solve, since there will be 
    # on on the following allreduce calls
    global_err = MPI.Allreduce(L₂norm, max, comm)
    next_Δt = MPI.Allreduce(local_next_Δt, min, comm)
  end

  return global_err, next_Δt
end

end # module