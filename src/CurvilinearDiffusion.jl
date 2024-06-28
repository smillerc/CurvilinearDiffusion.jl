module CurvilinearDiffusion

using UnPack
using CartesianDomains
using KernelAbstractions

include("timestep_control.jl")
using .TimeStepControl
export next_dt

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

include("pseudo_transient/PseudoTransientSolver.jl")
using .PseudoTransientScheme
export PseudoTransientSolver

include("conductivity.jl")
export update_conductivity!

include("validity_checks.jl")

include("nonlinear_thermal_conduction.jl")
export nonlinear_thermal_conduction_step!

include("vtk.jl")
using .VTKOutput
export save_vtk

end
