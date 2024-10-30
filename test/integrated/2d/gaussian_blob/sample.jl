using CairoMakie

begin
  function profile(x, t, Q)
    κ = 1.0

    (Q / sqrt(4π * κ * t)) * exp(-x^2 / (4κ * t))
  end

  x = -1:0.01:1
  x0 = 0.0
  fwhm = 0.025
  t = 0.01
  Q = @. exp(-(((x0 - x)^2) / fwhm))
  q = sum(Q[2:end] .* diff(x))
  @show q

  Qf = @. profile(x, t, q)

  f, ax, p = lines(x, Q; label="Q0")
  lines!(ax, x, Qf; label="Qf")
  axislegend()
  f
end
