
@inline function harmonic_mean(a, b)
  m = (2a * b) / (a + b)
  return m * isfinite(m)
end

@inline function weighted_harmonic_mean(a, w)
  m = sum(w) / sum(w ./ a)
  return m * isfinite(m)
end
@inline arithmetic_mean(a, b) = (a + b) / 2
@inline geometric_mean(a, b) = sqrt(a * b)
