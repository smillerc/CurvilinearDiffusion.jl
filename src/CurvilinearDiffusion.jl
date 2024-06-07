module CurvilinearDiffusion

using UnPack
using CartesianDomains
using KernelAbstractions

include("boundary_conditions/boundary_operators.jl")
using .BoundaryConditions
export DirichletBC, NeumannBC, PeriodicBC, applybc!, applybcs!, check_diffusivity_validity

include("implicit/ImplictSolver.jl")
using .ImplicitSchemeType
export ImplicitScheme, solve!, assemble!, initialize_coefficient_matrix

include("explicit/ADESolvers.jl")
using .ADESolvers
export AbstractADESolver, ADESolver, ADESolverNSweep, BlockADESolver
export solve!, validate_diffusivity

include("conductivity.jl")
export update_conductivity!

include("max_dt.jl")
using .TimeStepControl
export max_dt, maxxx_dt

include("validity_checks.jl")

include("nonlinear_thermal_conduction.jl")
export nonlinear_thermal_conduction_step!

include("vtk.jl")
using .VTKOutput
export save_vtk

end
