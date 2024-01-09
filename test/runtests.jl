using CurvilinearDiffusion
using TestItemRunner
using Test

@testset "CurvilinearDiffusion.jl" begin
  include("unit/test_edge_terms.jl")
  include("unit/test_metrics.jl")
end
