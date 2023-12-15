module CurvilinearDiffusion

include("ADESolver.jl")
using .ADESolverType
export ADESolver, solve!
export update_conductivity!, update_mesh_metrics!

include("averaging.jl")

include("max_dt.jl")
using .TimeStepControl
export max_dt

end
