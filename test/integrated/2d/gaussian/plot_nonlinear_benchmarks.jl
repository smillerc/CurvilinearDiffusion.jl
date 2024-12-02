using CairoMakie
using Unitful
using JSON3

function runtimes(fn)
  json_string = read(fn, String)
  data = JSON3.read(json_string)

  tot =
    Float64(data["inner_timers"]["nonlinear_thermal_conduction_step!"]["total_time_ns"]) *
    u"ns"
  n = data["inner_timers"]["nonlinear_thermal_conduction_step!"]["n_calls"]
  per_cycle = tot / n

  return tot |> u"s", per_cycle |> u"ms", n
end

prefix = "uniform_mesh_"
# resolution = [251, 501, 1001, 2001]
resolution = [251, 501]
cell_resolution = @. (resolution - 1)^2
# cell_resolution = @. 1 / (resolution - 1)#^2
krylov_gpu = [
  runtimes(
    "$(@__DIR__)/benchmark_results/$(prefix)nonlinear_GPU_timing_$(res)_implicit.json"
  ) for res in resolution
]

# krylov_cpu = [
#   runtimes(
#     "$(@__DIR__)/benchmark_results/$(prefix)nonlinear_CPU_timing_$(res)_implicit.json"
#   ) for res in resolution
# ]

pt_gpu = [
  runtimes(
    "$(@__DIR__)/benchmark_results/$(prefix)nonlinear_GPU_timing_$(res)_pseudo_transient.json",
  ) for res in resolution
]

# pt_cpu = [
#   runtimes(
#     "$(@__DIR__)/benchmark_results/$(prefix)nonlinear_CPU_timing_$(res)_pseudo_transient.json",
#   ) for res in resolution
# ]

update_theme!(; fontsize=20)
fig = Figure()
ax = Axis(
  fig[1, 1];
  yscale=log10,
  xscale=log10,
  xticks=[1e4, 1e5, 1e6, 1e7],
  yticks=[1e3, 1e4, 1e5, 1e6, 1e7],
  xminorticksvisible=true,
  xminorgridvisible=true,
  yminorticksvisible=true,
  yminorgridvisible=true,
  xminorticks=IntervalsBetween(9),
  yminorticks=IntervalsBetween(9),
  xlabel="Total Resolution",
  ylabel="Runtime [ms]",
  title="2D Nonlinear Thermal Conduction (Problem 4)",
)

palette = Makie.wong_colors()

runtime_idx = 1 # should it be 2 instead of 1??
scatterlines!(
  ax,
  cell_resolution,
  [ustrip(u"ms", c[runtime_idx]) for c in krylov_gpu];
  label="GMRES [GPU]",
  color=palette[1],
)

# scatterlines!(
#   ax,
#   cell_resolution,
#   [ustrip(u"ms", c[runtime_idx]) for c in krylov_cpu];
#   label="GMRES [CPU]",
#   linestyle=:dash,
#   color=palette[1],
# )

scatterlines!(
  ax,
  cell_resolution,
  [ustrip(u"ms", c[runtime_idx]) for c in pt_gpu];
  label="PT [GPU]",
  color=palette[2],
)

# scatterlines!(
#   ax,
#   cell_resolution,
#   [ustrip(u"ms", c[runtime_idx]) for c in pt_cpu];
#   label="PT [CPU]",
#   linestyle=:dash,
#   color=palette[2],
# )

lines!(ax, cell_resolution, cell_resolution; label="N", color=palette[3])
# lines!(
#   ax,
#   cell_resolution,
#   cell_resolution .* log10.(cell_resolution);
#   label="NlogN",
#   color=palette[4],
# )

# xlims!(ax, 10^4.2, 10^7.25)

# [round(typeof(1u"ms"), c[2]) for c in krylov_cpu]
# [round(typeof(1u"ms"), c[2]) for c in krylov_gpu]
# [round(typeof(1u"ms"), c[2]) for c in pt_cpu]
# [round(typeof(1u"ms"), c[2]) for c in pt_gpu]

Legend(fig[1, 2], ax)
display(fig)

save("$(@__DIR__)/nonlinear_benchmarks.png", fig)

# begin
#   dev = :CPU
#   gmres_low = CSV.File(
#     "/home/smil/.julia/dev/pt_diff_paper/test/integrated/2d/gaussian/benchmark_results/nonlinear_$(dev)_timing_501_implicit.csv",
#   )
#   gmres_med = CSV.File(
#     "/home/smil/.julia/dev/pt_diff_paper/test/integrated/2d/gaussian/benchmark_results/nonlinear_$(dev)_timing_1001_implicit.csv",
#   )
#   gmres_high = CSV.File(
#     "/home/smil/.julia/dev/pt_diff_paper/test/integrated/2d/gaussian/benchmark_results/nonlinear_$(dev)_timing_2001_implicit.csv",
#   )

#   PT_low = CSV.File(
#     "/home/smil/.julia/dev/pt_diff_paper/test/integrated/2d/gaussian/benchmark_results/nonlinear_$(dev)_timing_501_pseudo_transient.csv",
#   )
#   PT_med = CSV.File(
#     "/home/smil/.julia/dev/pt_diff_paper/test/integrated/2d/gaussian/benchmark_results/nonlinear_$(dev)_timing_1001_pseudo_transient.csv",
#   )
#   PT_high = CSV.File(
#     "/home/smil/.julia/dev/pt_diff_paper/test/integrated/2d/gaussian/benchmark_results/nonlinear_$(dev)_timing_2001_pseudo_transient.csv",
#   )

#   fig = Figure()
#   ax = Axis(fig[1, 1]; xlabel="Cycle", ylabel="Iterations / Cycle / nᵢ")

#   lines!(ax, gmres_low.iter_per_cycle ./ 500; label="GMRES low")
#   lines!(ax, gmres_med.iter_per_cycle ./ 1000; label="GMRES med")
#   lines!(ax, gmres_high.iter_per_cycle ./ 2000; label="GMRES high")

#   # lines!(ax, PT_low.iter_per_cycle; label="PT low")
#   # lines!(ax, PT_med.iter_per_cycle; label="PT medium")
#   # lines!(ax, PT_high.iter_per_cycle; label="PT high")

#   # lines!(ax, PT_low.iter_per_cycle ./ 500; label="PT low")
#   # lines!(ax, PT_med.iter_per_cycle ./ 1000; label="PT medium")
#   # lines!(ax, PT_high.iter_per_cycle ./ 2000; label="PT high")
#   axislegend(; position=:lt)
#   fig
# end
