using CSV
using Printf

fn = "$(@__DIR__)/heat_front_position.csv"
f = CSV.File(fn)

@show fn
x = f["Points:0"]
y = f["Points:1"]
z = f["Points:2"]

# scatter(x, y)

R = @. sqrt(x^2 + y^2 + z^2)

rmin, rmax = extrema(R)
# rave = 0.5 * (rmin + rmax)

N = length(R)
rave = sum(R) / N
@show rave
# @show abs(rmin - rmax) / rave

σ² = sum((R .- rave) .^ 2) / N

@printf "r̄ %.2e σ² %.2e" rave σ²

# @show σ²
nothing
