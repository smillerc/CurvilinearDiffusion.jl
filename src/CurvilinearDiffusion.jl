module CurvilinearDiffusion

include("partitioning.jl")
using .Partitioning

include("ImplictSolver.jl")
using .ImplicitSchemeType
export ImplicitScheme
export solve!, assemble_matrix!

include("ADESolvers.jl")
using .ADESolvers
export ADESolver
export BlockADESolver
export solve!, update_conductivity!, update_mesh_metrics!

include("max_dt.jl")
using .TimeStepControl
export max_dt

end
