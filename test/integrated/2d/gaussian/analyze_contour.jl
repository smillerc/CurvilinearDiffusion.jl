using CSV
using Printf
# f = CSV.File("$(@__DIR__)/t0_0.8contour.csv")
# f = CSV.File("$(@__DIR__)/t1_0.8contour.csv")

fn = "$(@__DIR__)/t0.csv"
# fn = "$(@__DIR__)/t40.csv"
f = CSV.File(fn)

@show fn
x = f["Points:0"]
y = f["Points:1"]
z = f["Points:2"]

# scatter(x, y)

R = @. sqrt(x^2 + y^2)

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

# R̂ = sum(R) / N

# θ = atan.(y, x)

# â = (2 / N) * sum(R .* cos.(θ))
# b̂ = (2 / N) * sum(R .* sin.(θ))

# Δ̂ = @. R - R̂ - â * cos(θ) - b̂ * sin(θ)
# @show norm(R̂ .- R)
# # @show norm(Δ̂)
# scatter(Δ̂)
# # lines(R)
