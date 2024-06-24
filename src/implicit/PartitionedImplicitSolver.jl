module PartitionedImplicitSchemeType

using CurvilinearGrids
using MPIHaloArrays
using MPI

struct PartitionedImplicitScheme{IS}
  local_scheme::IS
end

function PartitionedImplicitScheme()
  local_scheme = ImplicitScheme(
    mesh,
    bcs;
    direct_solve=false,
    face_conductivity::Symbol=:harmonic,
    T=Float64,
    backend=CPU(),
  )

  return PartitionedImplicitScheme(local_scheme)
end

function solve!(scheme::PartitionedImplicitScheme, mesh, u, Δt, comm; kwargs...)
  it = 0
  global_err = Inf
  next_Δt = -Inf
  tol = 1e-6

  # outer loop
  while it < it_max && global_err < tol

    # exchange edge terms
    updatehalo!(u)

    # inner solve
    stats, local_next_Δt = ImplicitSchemeType.solve!(
      scheme.local_scheme, mesh, u, Δt; kwargs...
    )

    # local convergence err
    L₂norm = last(stats.residuals)

    global_err = MPI.Allreduce(L₂norm, max, comm)
    next_Δt = MPI.Allreduce(local_next_Δt, min, comm)
  end

  return global_err, next_Δt
end

end # module