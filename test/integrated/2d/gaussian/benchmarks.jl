
using Unitful
using CairoMakie

begin # linear
  krylov_CPU = (
    (500, 1.45u"s", 50.1u"ms"), #
    (1000, 4.69u"s", 162.0u"ms"), #
    (4000, 61.0u"s", 2.10u"s"), #
  )

  krylov_GPU = (
    (500, 313.0u"ms", 10.8u"ms", 192), #
    (1000, 2.73u"s", 94.1u"ms"), #
    # (2000, 4.46u"s", 154.0u"ms"), #
    (4000, 11.3u"s", 390.0u"ms"), #
  )

  accelerated_PT_GPU = (
    (500, 313.0u"ms", 10.8u"ms", 192), #
    (1000, 964.0u"ms", 33.3u"ms", 236), #
    # (2000, 3.60u"s", 124.0u"ms", 236), #
    (4000, 17.4u"s", 601.0u"ms", 330), #
  )

  accelerated_PT_CPU = (
    (500, 3.55u"s", 122.0u"ms", 192), #
    (1000, 7.94u"s", 274.0u"ms", 236), #
    (4000, 73.6u"s", 2.54u"s", 330), #
  )

  case_idx = 2

  if case_idx == 2
    y_label = "Total Runtime [ms]"
  elseif case_idx == 3
    y_label = "Average Cycle Runtime [ms]"
  end

  fig = Figure()
  ax = Axis(
    fig[1, 1];
    yscale=log10,
    xscale=log10,
    ylabel=y_label,
    xlabel="Total Resolution",
    title="Linear Thermal Conduction (Problem 3)",
  )

  scatterlines!(
    ax,
    [case[1]^2 for case in krylov_CPU],
    [ustrip(u"ms", case[case_idx]) for case in krylov_CPU];
    label="Krylov solver [CPU]",
    linestyle=:dash,
  )
  scatterlines!(
    ax,
    [case[1]^2 for case in accelerated_PT_CPU],
    [ustrip(u"ms", case[case_idx]) for case in accelerated_PT_CPU];
    label="PT solver [CPU]",
    linestyle=:dash,
  )

  scatterlines!(
    ax,
    [case[1]^2 for case in krylov_GPU],
    [ustrip(u"ms", case[case_idx]) for case in krylov_GPU];
    label="Krylov solver [GPU]",
  )

  scatterlines!(
    ax,
    [case[1]^2 for case in accelerated_PT_GPU],
    [ustrip(u"ms", case[case_idx]) for case in accelerated_PT_GPU];
    label="PT solver [GPU]",
  )

  axislegend(; position=:lt)
  fig
end

begin # non-linear
  krylov_CPU = (
    (500, 1.45u"s", 50.1u"ms"), #
    (1000, 4.69u"s", 162.0u"ms"), #
    (4000, 61.0u"s", 2.10u"s"), #
  )

  #   krylov_GPU = (
  #     (500, 313.0u"ms", 10.8u"ms", 192), #
  #     (1000, 2.73u"s", 94.1u"ms"), #
  #     # (2000, 4.46u"s", 154.0u"ms"), #
  #     (4000, 11.3u"s", 390.0u"ms"), #
  #   )

  #   accelerated_PT_GPU = (
  #     (500, 313.0u"ms", 10.8u"ms", 192), #
  #     (1000, 964.0u"ms", 33.3u"ms", 236), #
  #     # (2000, 3.60u"s", 124.0u"ms", 236), #
  #     (4000, 17.4u"s", 601.0u"ms", 330), #
  #   )

  #   accelerated_PT_CPU = (
  #     (500, 3.55u"s", 122.0u"ms", 192), #
  #     (1000, 7.94u"s", 274.0u"ms", 236), #
  #     (4000, 73.6u"s", 2.54u"s", 330), #
  #   )

  case_idx = 2

  if case_idx == 2
    y_label = "Total Runtime [ms]"
  elseif case_idx == 3
    y_label = "Average Cycle Runtime [ms]"
  end

  fig = Figure()
  ax = Axis(
    fig[1, 1];
    yscale=log10,
    xscale=log10,
    ylabel=y_label,
    xlabel="Total Resolution",
    title="Linear Thermal Conduction (Problem 3)",
  )

  scatterlines!(
    ax,
    [case[1]^2 for case in krylov_CPU],
    [ustrip(u"ms", case[case_idx]) for case in krylov_CPU];
    label="Krylov solver [CPU]",
    linestyle=:dash,
  )
  scatterlines!(
    ax,
    [case[1]^2 for case in accelerated_PT_CPU],
    [ustrip(u"ms", case[case_idx]) for case in accelerated_PT_CPU];
    label="PT solver [CPU]",
    linestyle=:dash,
  )

  scatterlines!(
    ax,
    [case[1]^2 for case in krylov_GPU],
    [ustrip(u"ms", case[case_idx]) for case in krylov_GPU];
    label="Krylov solver [GPU]",
  )

  scatterlines!(
    ax,
    [case[1]^2 for case in accelerated_PT_GPU],
    [ustrip(u"ms", case[case_idx]) for case in accelerated_PT_GPU];
    label="PT solver [GPU]",
  )

  axislegend(; position=:lt)
  fig
end

# GPU 100x100
# cycle: 0 t: 0.0000e+00, Δt: 1.000e-06
#         rel error: 1.240e-06, abs err: 2.037e-07, 8
# cycle: 1 t: 1.0000e-06, Δt: 1.000e-06
#         rel error: 2.896e-07, abs err: 2.526e-08, 8
# cycle: 2 t: 2.0000e-06, Δt: 1.000e-06
#         rel error: 2.896e-07, abs err: 2.526e-08, 8
# cycle: 3 t: 3.0000e-06, Δt: 1.000e-06
#         rel error: 2.896e-07, abs err: 2.526e-08, 8
# cycle: 4 t: 4.0000e-06, Δt: 1.000e-06
#         rel error: 2.896e-07, abs err: 2.526e-08, 8
# cycle: 5 t: 5.0000e-06, Δt: 1.000e-06
#         rel error: 2.896e-07, abs err: 2.526e-08, 8
# cycle: 6 t: 6.0000e-06, Δt: 1.000e-06
#         rel error: 2.896e-07, abs err: 2.526e-08, 8
# cycle: 7 t: 7.0000e-06, Δt: 1.000e-06
#         rel error: 2.896e-07, abs err: 2.526e-08, 8
# cycle: 8 t: 8.0000e-06, Δt: 1.000e-06
#         rel error: 2.896e-07, abs err: 2.526e-08, 8
# cycle: 9 t: 9.0000e-06, Δt: 1.000e-06
#         rel error: 2.896e-07, abs err: 2.526e-08, 8
# cycle: 10 t: 1.0000e-05, Δt: 1.000e-06
#         rel error: 2.896e-07, abs err: 2.526e-08, 8
# ───────────────────────────────────────────────────────────────────────────────────────────────
#                                                       Time                    Allocations      
#                                              ───────────────────────   ────────────────────────
#               Tot / % measured:                   348ms /  99.9%           11.8MiB /  99.9%    

# Section                              ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────────────────────────────
# nonlinear_thermal_conduction_step!       11    348ms  100.0%  31.6ms   11.8MiB  100.0%  1.07MiB
#   update_conductivity!                   88    280ms   80.5%  3.18ms   6.33MiB   53.6%  73.6KiB
#   compute_update!                        88   13.3ms    3.8%   151μs   1.79MiB   15.1%  20.8KiB
#   applybcs! (u)                          88   11.5ms    3.3%   131μs    817KiB    6.8%  9.29KiB
#   update_residual!                       55   8.53ms    2.5%   155μs   1.12MiB    9.5%  20.8KiB
#   compute_flux!                          88   6.48ms    1.9%  73.6μs    559KiB    4.6%  6.35KiB
#   validate_scalar (u)                    22   6.44ms    1.8%   293μs    338KiB    2.8%  15.4KiB
#   update_iteration_params!               88   4.46ms    1.3%  50.6μs    234KiB    1.9%  2.65KiB
#   norm                                   55   4.34ms    1.2%  79.0μs    162KiB    1.3%  2.94KiB
#   validate_scalar (α)                    11   3.86ms    1.1%   351μs    171KiB    1.4%  15.5KiB
#   validate_scalar (source_term)          11   3.41ms    1.0%   310μs    169KiB    1.4%  15.3KiB
#   next_dt                                11   2.87ms    0.8%   261μs    123KiB    1.0%  11.2KiB
#   cutoff!                                11    196μs    0.1%  17.8μs   15.4KiB    0.1%  1.40KiB
# ───────────────────────────────────────────────────────────────────────────────────────────────

# GPU 500x500
# cycle: 0 t: 0.0000e+00, Δt: 1.000e-06
#         rel error: 6.403e-06, abs err: 3.169e-06, 16
# cycle: 1 t: 1.0000e-06, Δt: 1.000e-06
#         rel error: 6.126e-06, abs err: 1.892e-06, 14
# cycle: 2 t: 2.0000e-06, Δt: 1.000e-06
#         rel error: 6.126e-06, abs err: 1.892e-06, 14
# cycle: 3 t: 3.0000e-06, Δt: 1.000e-06
#         rel error: 6.126e-06, abs err: 1.892e-06, 14
# cycle: 4 t: 4.0000e-06, Δt: 1.000e-06
#         rel error: 6.126e-06, abs err: 1.892e-06, 14
# cycle: 5 t: 5.0000e-06, Δt: 1.000e-06
#         rel error: 6.126e-06, abs err: 1.892e-06, 14
# cycle: 6 t: 6.0000e-06, Δt: 1.000e-06
#         rel error: 6.126e-06, abs err: 1.892e-06, 14
# cycle: 7 t: 7.0000e-06, Δt: 1.000e-06
#         rel error: 6.126e-06, abs err: 1.892e-06, 14
# cycle: 8 t: 8.0000e-06, Δt: 1.000e-06
#         rel error: 6.126e-06, abs err: 1.892e-06, 14
# cycle: 9 t: 9.0000e-06, Δt: 1.000e-06
#         rel error: 6.126e-06, abs err: 1.892e-06, 14
# cycle: 10 t: 1.0000e-05, Δt: 1.000e-06
#         rel error: 6.126e-06, abs err: 1.892e-06, 14
# ───────────────────────────────────────────────────────────────────────────────────────────────
#                                                       Time                    Allocations      
#                                              ───────────────────────   ────────────────────────
#               Tot / % measured:                   505ms /  99.9%           15.6MiB /  99.9%    

# Section                              ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────────────────────────────
# nonlinear_thermal_conduction_step!       11    504ms  100.0%  45.9ms   15.6MiB  100.0%  1.42MiB
#   update_conductivity!                  156    287ms   56.8%  1.84ms   6.48MiB   41.6%  42.5KiB
#   compute_update!                       156   81.4ms   16.1%   522μs   3.16MiB   20.3%  20.8KiB
#   update_residual!                       89   42.3ms    8.4%   475μs   1.81MiB   11.6%  20.8KiB
#   compute_flux!                         156   27.7ms    5.5%   177μs   0.96MiB    6.1%  6.29KiB
#   applybcs! (u)                         156   18.8ms    3.7%   120μs   1.41MiB    9.1%  9.28KiB
#   update_iteration_params!              156   18.2ms    3.6%   117μs    408KiB    2.6%  2.62KiB
#   validate_scalar (u)                    22   7.88ms    1.6%   358μs    474KiB    3.0%  21.6KiB
#   norm                                   89   7.43ms    1.5%  83.5μs    261KiB    1.6%  2.93KiB
#   validate_scalar (α)                    11   4.50ms    0.9%   409μs    239KiB    1.5%  21.7KiB
#   validate_scalar (source_term)          11   4.03ms    0.8%   367μs    237KiB    1.5%  21.6KiB
#   next_dt                                11   2.58ms    0.5%   235μs    123KiB    0.8%  11.2KiB
#   cutoff!                                11    206μs    0.0%  18.8μs   15.4KiB    0.1%  1.40KiB
# ───────────────────────────────────────────────────────────────────────────────────────────────

# GPU 1000x1000
# cycle: 0 t: 0.0000e+00, Δt: 1.000e-06
#         rel error: 6.304e-06, abs err: 4.011e-06, 26
# cycle: 1 t: 1.0000e-06, Δt: 1.000e-06
#         rel error: 7.090e-06, abs err: 3.152e-06, 22
# cycle: 2 t: 2.0000e-06, Δt: 1.000e-06
#         rel error: 7.090e-06, abs err: 3.152e-06, 22
# cycle: 3 t: 3.0000e-06, Δt: 1.000e-06
#         rel error: 7.090e-06, abs err: 3.152e-06, 22
# cycle: 4 t: 4.0000e-06, Δt: 1.000e-06
#         rel error: 7.090e-06, abs err: 3.152e-06, 22
# cycle: 5 t: 5.0000e-06, Δt: 1.000e-06
#         rel error: 7.090e-06, abs err: 3.152e-06, 22
# cycle: 6 t: 6.0000e-06, Δt: 1.000e-06
#         rel error: 7.090e-06, abs err: 3.152e-06, 22
# cycle: 7 t: 7.0000e-06, Δt: 1.000e-06
#         rel error: 7.090e-06, abs err: 3.151e-06, 22
# cycle: 8 t: 8.0000e-06, Δt: 1.000e-06
#         rel error: 7.090e-06, abs err: 3.151e-06, 22
# cycle: 9 t: 9.0000e-06, Δt: 1.000e-06
#         rel error: 7.090e-06, abs err: 3.151e-06, 22
# cycle: 10 t: 1.0000e-05, Δt: 1.000e-06
#         rel error: 7.090e-06, abs err: 3.151e-06, 22
# ───────────────────────────────────────────────────────────────────────────────────────────────
#                                                       Time                    Allocations      
#                                              ───────────────────────   ────────────────────────
#               Tot / % measured:                   1.24s / 100.0%           20.4MiB /  99.9%    

# Section                              ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────────────────────────────
# nonlinear_thermal_conduction_step!       11    1.23s  100.0%   112ms   20.4MiB  100.0%  1.85MiB
#   compute_update!                       246    386ms   31.2%  1.57ms   5.02MiB   24.6%  20.9KiB
#   update_conductivity!                  246    293ms   23.7%  1.19ms   6.68MiB   32.8%  27.8KiB
#   update_residual!                      134    198ms   16.0%  1.48ms   2.73MiB   13.4%  20.9KiB
#   compute_flux!                         246    148ms   12.0%   600μs   1.56MiB    7.7%  6.51KiB
#   update_iteration_params!              246   94.1ms    7.6%   383μs    670KiB    3.2%  2.72KiB
#   applybcs! (u)                         246   55.0ms    4.5%   224μs   2.23MiB   10.9%  9.28KiB
#   norm                                  134   19.3ms    1.6%   144μs    392KiB    1.9%  2.93KiB
#   validate_scalar (u)                    22   14.8ms    1.2%   671μs    474KiB    2.3%  21.6KiB
#   validate_scalar (α)                    11   7.30ms    0.6%   663μs    239KiB    1.1%  21.7KiB
#   validate_scalar (source_term)          11   6.78ms    0.5%   617μs    237KiB    1.1%  21.5KiB
#   next_dt                                11   5.17ms    0.4%   470μs    123KiB    0.6%  11.2KiB
#   cutoff!                                11    392μs    0.0%  35.6μs   15.4KiB    0.1%  1.40KiB
# ───────────────────────────────────────────────────────────────────────────────────────────────

# GPU 4000x4000
# cycle: 0 t: 0.0000e+00, Δt: 1.000e-06
#         rel error: 9.386e-06, abs err: 7.182e-06, 82
# cycle: 1 t: 1.0000e-06, Δt: 1.000e-06
#         rel error: 8.833e-06, abs err: 5.830e-06, 68
# cycle: 2 t: 2.0000e-06, Δt: 1.000e-06
#         rel error: 8.833e-06, abs err: 5.830e-06, 68
# cycle: 3 t: 3.0000e-06, Δt: 1.000e-06
#         rel error: 8.833e-06, abs err: 5.830e-06, 68
# cycle: 4 t: 4.0000e-06, Δt: 1.000e-06
#         rel error: 8.833e-06, abs err: 5.830e-06, 68
# cycle: 5 t: 5.0000e-06, Δt: 1.000e-06
#         rel error: 8.833e-06, abs err: 5.830e-06, 68
# cycle: 6 t: 6.0000e-06, Δt: 1.000e-06
#         rel error: 8.833e-06, abs err: 5.829e-06, 68
# cycle: 7 t: 7.0000e-06, Δt: 1.000e-06
#         rel error: 8.833e-06, abs err: 5.829e-06, 68
# cycle: 8 t: 8.0000e-06, Δt: 1.000e-06
#         rel error: 8.833e-06, abs err: 5.829e-06, 68
# cycle: 9 t: 9.0000e-06, Δt: 1.000e-06
#         rel error: 8.833e-06, abs err: 5.829e-06, 68
# cycle: 10 t: 1.0000e-05, Δt: 1.000e-06
#         rel error: 8.833e-06, abs err: 5.829e-06, 68
# ───────────────────────────────────────────────────────────────────────────────────────────────
#                                                       Time                    Allocations      
#                                              ───────────────────────   ────────────────────────
#               Tot / % measured:                   34.5s / 100.0%           47.4MiB / 100.0%    

# Section                              ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────────────────────────────
# nonlinear_thermal_conduction_step!       11    34.5s  100.0%   3.13s   47.4MiB  100.0%  4.31MiB
#   compute_update!                       762    15.6s   45.2%  20.5ms   15.5MiB   32.8%  20.9KiB
#   update_residual!                      392    7.45s   21.6%  19.0ms   7.99MiB   16.9%  20.9KiB
#   compute_flux!                         762    5.47s   15.9%  7.17ms   4.85MiB   10.2%  6.52KiB
#   update_iteration_params!              762    4.25s   12.3%  5.58ms   2.04MiB    4.3%  2.74KiB
#   applybcs! (u)                         762    960ms    2.8%  1.26ms   6.91MiB   14.6%  9.28KiB
#   update_conductivity!                  762    355ms    1.0%   466μs   7.83MiB   16.5%  10.5KiB
#   norm                                  392    234ms    0.7%   598μs   1.12MiB    2.4%  2.92KiB
#   validate_scalar (u)                    22   39.5ms    0.1%  1.79ms    474KiB    1.0%  21.6KiB
#   validate_scalar (α)                    11   35.7ms    0.1%  3.24ms    239KiB    0.5%  21.7KiB
#   next_dt                                11   25.8ms    0.1%  2.34ms    124KiB    0.3%  11.3KiB
#   validate_scalar (source_term)          11   15.5ms    0.0%  1.41ms    237KiB    0.5%  21.5KiB
#   cutoff!                                11    736μs    0.0%  66.9μs   15.4KiB    0.0%  1.40KiB
# ───────────────────────────────────────────────────────────────────────────────────────────────

## Krylov GPU

# 100x100
# cycle: 0 t: 0.0000e+00, Δt: 1.000e-06
#         Krylov stats: L₂: 1.1e-16, iterations: 2
# cycle: 1 t: 1.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-14, iterations: 1
# cycle: 2 t: 2.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-14, iterations: 1
# cycle: 3 t: 3.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-14, iterations: 1
# cycle: 4 t: 4.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-14, iterations: 1
# cycle: 5 t: 5.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-14, iterations: 1
# cycle: 6 t: 6.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-14, iterations: 1
# cycle: 7 t: 7.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-14, iterations: 1
# cycle: 8 t: 8.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-14, iterations: 1
# cycle: 9 t: 9.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-14, iterations: 1
# cycle: 10 t: 1.0000e-05, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-14, iterations: 1
# ───────────────────────────────────────────────────────────────────────────────────────────────
#                                                       Time                    Allocations      
#                                              ───────────────────────   ────────────────────────
#               Tot / % measured:                  97.3ms /  99.6%           1.26MiB /  98.7%    

# Section                              ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────────────────────────────
# nonlinear_thermal_conduction_step!       11   96.9ms  100.0%  8.81ms   1.25MiB  100.0%   116KiB
#   solve!                                 11   85.9ms   88.6%  7.81ms    629KiB   49.4%  57.2KiB
#     linear solve                         11   75.9ms   78.3%  6.90ms    395KiB   31.0%  35.9KiB
#     next_dt                              11   7.94ms    8.2%   722μs   66.3KiB    5.2%  6.03KiB
#     assembly                             11    684μs    0.7%  62.2μs    134KiB   10.5%  12.2KiB
#     preconditioner                        1   69.2μs    0.1%  69.2μs      832B    0.1%     832B
#   validate_diffusivity                   11   5.86ms    6.0%   533μs    351KiB   27.5%  31.9KiB
#   validate_source_term                   11   3.23ms    3.3%   294μs    169KiB   13.3%  15.4KiB
#   applybc!                               22   1.11ms    1.1%  50.5μs   68.1KiB    5.3%  3.09KiB
#   update_conductivity!                   11    541μs    0.6%  49.2μs   43.5KiB    3.4%  3.95KiB
# ───────────────────────────────────────────────────────────────────────────────────────────────

# 500x500
# cycle: 0 t: 0.0000e+00, Δt: 1.000e-06
#         Krylov stats: L₂: 9.2e-12, iterations: 2
# cycle: 1 t: 1.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 4.1e-12, iterations: 1
# cycle: 2 t: 2.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 4.1e-12, iterations: 1
# cycle: 3 t: 3.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 4.1e-12, iterations: 1
# cycle: 4 t: 4.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 4.1e-12, iterations: 1
# cycle: 5 t: 5.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 4.1e-12, iterations: 1
# cycle: 6 t: 6.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 4.1e-12, iterations: 1
# cycle: 7 t: 7.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 4.1e-12, iterations: 1
# cycle: 8 t: 8.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 4.1e-12, iterations: 1
# cycle: 9 t: 9.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 4.1e-12, iterations: 1
# cycle: 10 t: 1.0000e-05, Δt: 1.000e-06
#         Krylov stats: L₂: 4.1e-12, iterations: 1
# ───────────────────────────────────────────────────────────────────────────────────────────────
#                                                       Time                    Allocations      
#                                              ───────────────────────   ────────────────────────
#               Tot / % measured:                   475ms /  99.9%           1.27MiB /  98.7%    

# Section                              ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────────────────────────────
# nonlinear_thermal_conduction_step!       11    474ms  100.0%  43.1ms   1.26MiB  100.0%   117KiB
#   solve!                                 11    461ms   97.3%  41.9ms    636KiB   49.5%  57.8KiB
#     linear solve                         11    400ms   84.4%  36.4ms    397KiB   30.9%  36.1KiB
#     next_dt                              11   55.1ms   11.6%  5.01ms   69.0KiB    5.4%  6.27KiB
#     assembly                             11    869μs    0.2%  79.0μs    135KiB   10.5%  12.2KiB
#     preconditioner                        1    138μs    0.0%   138μs      832B    0.1%     832B
#   validate_diffusivity                   11   6.74ms    1.4%   613μs    352KiB   27.4%  32.0KiB
#   validate_source_term                   11   3.51ms    0.7%   319μs    171KiB   13.3%  15.5KiB
#   applybc!                               22   1.64ms    0.3%  74.4μs   68.6KiB    5.3%  3.12KiB
#   update_conductivity!                   11    858μs    0.2%  78.0μs   43.1KiB    3.4%  3.92KiB
# ───────────────────────────────────────────────────────────────────────────────────────────────

# 1000x1000
# cycle: 0 t: 0.0000e+00, Δt: 1.000e-06
#         Krylov stats: L₂: 9.1e-10, iterations: 2
# cycle: 1 t: 1.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-11, iterations: 1
# cycle: 2 t: 2.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-11, iterations: 1
# cycle: 3 t: 3.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-11, iterations: 1
# cycle: 4 t: 4.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-11, iterations: 1
# cycle: 5 t: 5.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-11, iterations: 1
# cycle: 6 t: 6.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-11, iterations: 1
# cycle: 7 t: 7.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-11, iterations: 1
# cycle: 8 t: 8.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-11, iterations: 1
# cycle: 9 t: 9.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-11, iterations: 1
# cycle: 10 t: 1.0000e-05, Δt: 1.000e-06
#         Krylov stats: L₂: 2.9e-11, iterations: 1
# ───────────────────────────────────────────────────────────────────────────────────────────────
#                                                       Time                    Allocations      
#                                              ───────────────────────   ────────────────────────
#               Tot / % measured:                   1.09s /  99.9%           1.47MiB /  98.9%    

# Section                              ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────────────────────────────
# nonlinear_thermal_conduction_step!       11    1.09s  100.0%  99.0ms   1.46MiB  100.0%   136KiB
#   solve!                                 11    1.07s   98.2%  97.3ms    636KiB   42.7%  57.8KiB
#     linear solve                         11    905ms   83.1%  82.3ms    397KiB   26.6%  36.1KiB
#     next_dt                              11    150ms   13.7%  13.6ms   69.0KiB    4.6%  6.27KiB
#     assembly                             11    988μs    0.1%  89.9μs    135KiB    9.0%  12.2KiB
#     preconditioner                        1    149μs    0.0%   149μs      832B    0.1%     832B
#   validate_diffusivity                   11   10.5ms    1.0%   950μs    489KiB   32.8%  44.4KiB
#   validate_source_term                   11   5.20ms    0.5%   473μs    239KiB   16.0%  21.7KiB
#   update_conductivity!                   11   1.68ms    0.2%   153μs   44.3KiB    3.0%  4.03KiB
#   applybc!                               22   1.48ms    0.1%  67.3μs   68.6KiB    4.6%  3.12KiB
# ───────────────────────────────────────────────────────────────────────────────────────────────

# 4000x4000
# cycle: 0 t: 0.0000e+00, Δt: 1.000e-06
#         Krylov stats: L₂: 4.2e-10, iterations: 4
# cycle: 1 t: 1.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 4.6e-10, iterations: 1
# cycle: 2 t: 2.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 4.6e-10, iterations: 1
# cycle: 3 t: 3.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 4.6e-10, iterations: 1
# cycle: 4 t: 4.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 4.6e-10, iterations: 1
# cycle: 5 t: 5.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 4.6e-10, iterations: 1
# cycle: 6 t: 6.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 4.6e-10, iterations: 1
# cycle: 7 t: 7.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 4.6e-10, iterations: 1
# cycle: 8 t: 8.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 4.6e-10, iterations: 1
# cycle: 9 t: 9.0000e-06, Δt: 1.000e-06
#         Krylov stats: L₂: 4.6e-10, iterations: 1
# cycle: 10 t: 1.0000e-05, Δt: 1.000e-06
#         Krylov stats: L₂: 4.6e-10, iterations: 1
# ───────────────────────────────────────────────────────────────────────────────────────────────
#                                                       Time                    Allocations      
#                                              ───────────────────────   ────────────────────────
#               Tot / % measured:                   23.4s /  98.1%           1.27GiB /  97.9%    

# Section                              ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────────────────────────────
# nonlinear_thermal_conduction_step!       11    22.9s  100.0%   2.08s   1.25GiB  100.0%   116MiB
#   solve!                                 11    8.87s   38.7%   806ms    176MiB   13.7%  16.0MiB
#     linear solve                         11    4.96s   21.6%   451ms   25.0MiB    2.0%  2.27MiB
#     next_dt                              11    1.99s    8.7%   181ms   64.4MiB    5.0%  5.85MiB
#     assembly                             11    920ms    4.0%  83.6ms   62.7MiB    4.9%  5.70MiB
#     preconditioner                        1    110μs    0.0%   110μs      832B    0.0%     832B
#   validate_source_term                   11    1.54s    6.7%   140ms    104MiB    8.1%  9.46MiB
#   validate_diffusivity                   11    864ms    3.8%  78.5ms   17.6MiB    1.4%  1.60MiB
#   applybc!                               22    125ms    0.5%  5.67ms   10.5MiB    0.8%   488KiB
#   update_conductivity!                   11   56.1ms    0.2%  5.10ms   47.0KiB    0.0%  4.28KiB
# ───────────────────────────────────────────────────────────────────────────────────────────────

## nonlinear with source term

# Krylov GPU 500x500
# ───────────────────────────────────────────────────────────────────────────────────────────────
#                                                       Time                    Allocations      
#                                              ───────────────────────   ────────────────────────
#               Tot / % measured:                   42.9s / 100.0%            115MiB /  99.7%    

# Section                              ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────────────────────────────
# nonlinear_thermal_conduction_step!      270    42.6s   99.3%   158ms   48.5MiB   42.3%   184KiB
#   solve!                                270    42.3s   98.6%   157ms   33.2MiB   28.9%   126KiB
#     linear solve                        270    40.9s   95.3%   151ms   27.4MiB   23.9%   104KiB
#     next_dt                             270    1.27s    3.0%  4.72ms   1.61MiB    1.4%  6.12KiB
#     assembly                            270   18.3ms    0.0%  67.6μs   3.28MiB    2.9%  12.4KiB
#     preconditioner                       17   1.06ms    0.0%  62.1μs   13.8KiB    0.0%     832B
#   validate_diffusivity                  270    147ms    0.3%   543μs   8.41MiB    7.3%  31.9KiB
#   validate_source_term                  270   77.9ms    0.2%   289μs   4.05MiB    3.5%  15.4KiB
#   applybc!                              540   26.4ms    0.1%  48.9μs   1.63MiB    1.4%  3.09KiB
#   update_conductivity!                  270   21.0ms    0.0%  77.9μs   1.03MiB    0.9%  3.92KiB
# save_vtk                                  1    307ms    0.7%   307ms   66.3MiB   57.7%  66.3MiB
# ───────────────────────────────────────────────────────────────────────────────────────────────

# PT GPU 500x500
# ───────────────────────────────────────────────────────────────────────────────────────────────
#                                                       Time                    Allocations      
#                                              ───────────────────────   ────────────────────────
#               Tot / % measured:                   16.8s / 100.0%            829MiB / 100.0%    

# Section                              ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────────────────────────────
# nonlinear_thermal_conduction_step!      251    16.1s   95.8%  64.0ms    727MiB   87.7%  2.90MiB
#   compute_update!                     15.2k    7.29s   43.4%   478μs    313MiB   37.7%  21.0KiB
#   update_residual!                    7.87k    3.47s   20.7%   441μs    162MiB   19.5%  21.0KiB
#   update_iteration_params!            15.2k    1.62s    9.7%   106μs   38.8MiB    4.7%  2.61KiB
#   compute_flux!                       15.2k    1.60s    9.5%   105μs   93.5MiB   11.3%  6.28KiB
#   applybcs! (u)                       15.2k    683ms    4.1%  44.8μs   46.0MiB    5.6%  3.09KiB
#   norm                                7.87k    581ms    3.5%  73.8μs   22.4MiB    2.7%  2.92KiB
#   update_conductivity!                15.2k    353ms    2.1%  23.2μs   34.0MiB    4.1%  2.28KiB
#   validate_scalar (u)                   502    133ms    0.8%   265μs   7.53MiB    0.9%  15.4KiB
#   validate_scalar (α)                   251   70.7ms    0.4%   282μs   3.76MiB    0.5%  15.4KiB
#   validate_scalar (source_term)         251   62.1ms    0.4%   247μs   3.77MiB    0.5%  15.4KiB
#   next_dt                               251   41.2ms    0.2%   164μs   1.55MiB    0.2%  6.31KiB
#   cutoff!                               251   4.99ms    0.0%  19.9μs    337KiB    0.0%  1.34KiB
# save_vtk                                  1    700ms    4.2%   700ms    102MiB   12.3%   102MiB
# ───────────────────────────────────────────────────────────────────────────────────────────────

# PT GPU 1000x1000
# ───────────────────────────────────────────────────────────────────────────────────────────────
#                                                       Time                    Allocations      
#                                              ───────────────────────   ────────────────────────
#               Tot / % measured:                   96.9s / 100.0%           1.77GiB / 100.0%    

# Section                              ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────────────────────────────
# nonlinear_thermal_conduction_step!      251    94.1s   97.2%   375ms   1.35GiB   76.0%  5.49MiB
#   compute_update!                     28.8k    44.0s   45.4%  1.53ms    594MiB   32.7%  21.1KiB
#   update_residual!                    14.6k    21.2s   21.9%  1.45ms    302MiB   16.7%  21.1KiB
#   update_iteration_params!            28.8k    11.1s   11.5%   386μs   76.4MiB    4.2%  2.72KiB
#   compute_flux!                       28.8k    9.90s   10.2%   344μs    183MiB   10.1%  6.50KiB
#   applybcs! (u)                       28.8k    2.78s    2.9%  96.8μs   86.9MiB    4.8%  3.09KiB
#   norm                                14.6k    2.16s    2.2%   148μs   41.7MiB    2.3%  2.92KiB
#   update_conductivity!                28.8k    1.67s    1.7%  58.0μs   70.2MiB    3.9%  2.50KiB
#   validate_scalar (u)                   502    326ms    0.3%   649μs   10.6MiB    0.6%  21.5KiB
#   validate_scalar (α)                   251    170ms    0.2%   676μs   5.28MiB    0.3%  21.6KiB
#   validate_scalar (source_term)         251    157ms    0.2%   627μs   5.28MiB    0.3%  21.6KiB
#   next_dt                               251   80.7ms    0.1%   321μs   1.60MiB    0.1%  6.52KiB
#   cutoff!                               251   9.63ms    0.0%  38.4μs    338KiB    0.0%  1.35KiB
# save_vtk                                  1    2.72s    2.8%   2.72s    435MiB   24.0%   435MiB
# ───────────────────────────────────────────────────────────────────────────────────────────────

# Krylov GPU 1000x1000
# ───────────────────────────────────────────────────────────────────────────────────────────────
#                                                       Time                    Allocations      
#                                              ───────────────────────   ────────────────────────
#               Tot / % measured:                    163s / 100.0%            348MiB /  99.9%    

# Section                              ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────────────────────────────
# nonlinear_thermal_conduction_step!      278     162s   99.1%   581ms   85.0MiB   24.4%   313KiB
#   solve!                                278     160s   98.3%   576ms   51.2MiB   14.7%   189KiB
#     linear solve                        278     156s   95.7%   561ms   44.4MiB   12.8%   164KiB
#     next_dt                             278    3.79s    2.3%  13.6ms   1.65MiB    0.5%  6.06KiB
#     assembly                            278   25.2ms    0.0%  90.6μs   3.38MiB    1.0%  12.4KiB
#     preconditioner                      121   10.4ms    0.0%  85.9μs   98.3KiB    0.0%     832B
#   validate_diffusivity                  278    258ms    0.2%   927μs   12.0MiB    3.5%  44.3KiB
#   validate_source_term                  278    135ms    0.1%   484μs   5.85MiB    1.7%  21.5KiB
#   update_conductivity!                  278   46.1ms    0.0%   166μs   1.09MiB    0.3%  4.03KiB
#   applybc!                              556   36.3ms    0.0%  65.2μs   1.68MiB    0.5%  3.09KiB
# save_vtk                                  1    1.42s    0.9%   1.42s    263MiB   75.6%   263MiB
# ───────────────────────────────────────────────────────────────────────────────────────────────

# Krylov CPU 1000x1000
# ───────────────────────────────────────────────────────────────────────────────────────────────
#                                                       Time                    Allocations      
#                                              ───────────────────────   ────────────────────────
#               Tot / % measured:                   1009s / 100.0%           2.24GiB / 100.0%    

# Section                              ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────────────────────────────
# nonlinear_thermal_conduction_step!      278    1007s   99.9%   3.62s   2.02GiB   90.2%  7.43MiB
#   solve!                                278    1000s   99.1%   3.60s   1.52GiB   68.0%  5.60MiB
#     linear solve                        278     967s   95.8%   3.48s   1.47GiB   65.9%  5.43MiB
#     preconditioner                      149    16.4s    1.6%   110ms     0.00B    0.0%    0.00B
#     assembly                            278    12.0s    1.2%  43.1ms   37.5MiB    1.6%   138KiB
#     next_dt                             278    3.41s    0.3%  12.3ms      960B    0.0%    3.45B
#   validate_diffusivity                  278    2.43s    0.2%  8.73ms    305MiB   13.3%  1.10MiB
#   update_conductivity!                  278    808ms    0.1%  2.91ms   4.63MiB    0.2%  17.0KiB
#   validate_source_term                  278    333ms    0.0%  1.20ms   39.1MiB    1.7%   144KiB
#   applybc!                              556   14.8ms    0.0%  26.6μs     0.00B    0.0%    0.00B
# save_vtk                                  1    1.24s    0.1%   1.24s    225MiB    9.8%   225MiB
# ───────────────────────────────────────────────────────────────────────────────────────────────

# Krylov CPU 500x500
# ───────────────────────────────────────────────────────────────────────────────────────────────
#                                                       Time                    Allocations      
#                                              ───────────────────────   ────────────────────────
#               Tot / % measured:                    129s / 100.0%            345MiB /  99.9%    

# Section                              ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────────────────────────────
# nonlinear_thermal_conduction_step!      270     129s   99.7%   477ms    288MiB   83.6%  1.07MiB
#   solve!                                270     128s   98.9%   473ms    191MiB   55.3%   724KiB
#     linear solve                        270     118s   91.6%   438ms    181MiB   52.4%   686KiB
#     preconditioner                      137    4.74s    3.7%  34.6ms     0.00B    0.0%    0.00B
#     assembly                            270    3.26s    2.5%  12.1ms   5.57MiB    1.6%  21.1KiB
#     next_dt                             270    979ms    0.8%  3.63ms     0.00B    0.0%    0.00B
#   validate_diffusivity                  270    649ms    0.5%  2.40ms   78.8MiB   22.8%   299KiB
#   update_conductivity!                  270    258ms    0.2%   956μs   4.49MiB    1.3%  17.0KiB
#   validate_source_term                  270   91.5ms    0.1%   339μs   13.8MiB    4.0%  52.3KiB
#   applybc!                              540   6.60ms    0.0%  12.2μs     0.00B    0.0%    0.00B
# save_vtk                                  1    351ms    0.3%   351ms   56.7MiB   16.4%  56.7MiB
# ───────────────────────────────────────────────────────────────────────────────────────────────

# PT CPU 500x500
# ───────────────────────────────────────────────────────────────────────────────────────────────
#                                                       Time                    Allocations      
#                                              ───────────────────────   ────────────────────────
#               Tot / % measured:                    258s / 100.0%           1.21GiB / 100.0%    

# Section                              ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────────────────────────────
# nonlinear_thermal_conduction_step!      251     257s   99.7%   1.02s   1.14GiB   94.0%  4.64MiB
#   compute_update!                     15.2k     139s   53.8%  9.09ms    308MiB   24.9%  20.7KiB
#   update_residual!                    7.87k    41.1s   16.0%  5.23ms    159MiB   12.8%  20.7KiB
#   update_iteration_params!            15.2k    40.7s   15.8%  2.67ms    205MiB   16.5%  13.8KiB
#   compute_flux!                       15.2k    31.6s   12.3%  2.07ms    413MiB   33.3%  27.7KiB
#   update_conductivity!                15.2k    1.16s    0.5%  76.2μs   3.49MiB    0.3%     240B
#   norm                                7.87k    566ms    0.2%  71.9μs   1.32MiB    0.1%     176B
#   next_dt                               251    376ms    0.1%  1.50ms     0.00B    0.0%    0.00B
#   applybcs! (u)                       15.2k    154ms    0.1%  10.1μs     0.00B    0.0%    0.00B
#   validate_scalar (u)                   502    108ms    0.0%   214μs   25.6MiB    2.1%  52.3KiB
#   validate_scalar (α)                   251   69.8ms    0.0%   278μs   12.8MiB    1.0%  52.3KiB
#   validate_scalar (source_term)         251   65.1ms    0.0%   259μs   12.8MiB    1.0%  52.3KiB
#   cutoff!                               251   10.4ms    0.0%  41.3μs     0.00B    0.0%    0.00B
# save_vtk                                  1    732ms    0.3%   732ms   74.9MiB    6.0%  74.9MiB
# ───────────────────────────────────────────────────────────────────────────────────────────────