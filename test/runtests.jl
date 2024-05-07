using CurvilinearDiffusion
using CurvilinearGrids
using Test

@testset "CurvilinearDiffusion.jl" verbose = true begin
  # include("unit/test_edge_terms.jl")
  include("unit/test_bc.jl")
  include("unit/test_implicit_scheme.jl")
end
