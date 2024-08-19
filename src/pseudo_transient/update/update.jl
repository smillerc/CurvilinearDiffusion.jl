
include("orthogonal_mesh_update_gpu.jl")
include("orthogonal_mesh_update_cpu.jl")
include("nonorthogonal_mesh_update_gpu.jl")
include("nonorthogonal_mesh_update_cpu.jl")

function compute_update!(solver::PseudoTransientSolver{N,T,BE}, mesh, Δt) where {N,T,BE}
  if mesh.is_orthogonal
    update_orthogonal!(solver, mesh, Δt)
  else
    update_nonorthogonal!(solver, mesh, Δt)
  end
end