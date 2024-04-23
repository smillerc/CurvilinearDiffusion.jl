using CurvilinearDiffusion
using Test
using StaticArrays
using BenchmarkTools

@testset "CurvilinearDiffusion.jl" begin
  include("unit/test_inner_operator.jl")
end
