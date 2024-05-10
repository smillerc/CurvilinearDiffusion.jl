module CurvilinearDiffusion

using UnPack
using CartesianDomains
using KernelAbstractions

include("implicit/ImplictSolver.jl")
using .ImplicitSchemeType
export ImplicitScheme, solve!, assemble!, initialize_coefficient_matrix
export DirichletBC, NeumannBC, PeriodicBC, applybc!, applybcs!, check_diffusivity_validity

include("nonlinear_thermal_conduction.jl")
export nonlinear_thermal_conduction_step!

# include("explicit/ADESolvers.jl")

include("conductivity.jl")
export update_conductivity!

include("max_dt.jl")
using .TimeStepControl
export max_dt

include("vtk.jl")
using .VTKOutput
export save_vtk

end
