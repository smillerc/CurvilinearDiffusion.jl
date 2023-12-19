
@inline harmonic_mean(a, b) = (2a * b) / (a + b)
@inline function weighted_harmonic_mean(a, w)
  m = sum(w) / sum(w ./ a)
  return m * isfinite(m)
end
@inline arithmetic_mean(a, b) = (a + b) / 2
@inline geometric_mean(a, b) = sqrt(a * b)
