module CurvilinearDiffusion

using UnPack
using CartesianDomains
using KernelAbstractions

include("implicit/ImplictSolver.jl")
using .ImplicitSchemeType
export ImplicitScheme
export solve!

include("conductivity.jl")
export update_conductivity!

include("max_dt.jl")
using .TimeStepControl
export max_dt

include("vtk.jl")
using .VTKOutput
export save_vtk

end
