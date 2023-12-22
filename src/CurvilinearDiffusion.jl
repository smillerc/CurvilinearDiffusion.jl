module CurvilinearDiffusion

include("partitioning.jl")
using .Partitioning

include("ADESolvers.jl")
using .ADESolvers
export ADESolver
export BlockADESolver
export solve!, update_conductivity!, update_mesh_metrics!

include("max_dt.jl")
using .TimeStepControl
export max_dt

end
