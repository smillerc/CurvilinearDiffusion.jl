function validate_scalar(
  α::AbstractArray{T,N}, domain, nhalo::Int, name::Symbol; enforce_positivity=false
) where {T,N}
  full = expand(domain, nhalo)
  α_domain = @view α[domain]

  if enforce_positivity
    # domain_valid = (all(isfinite.(α_domain)) && all(map(x -> x > 0, α_domain)))
    domain_valid = (all(isfinite.(α_domain)) && all(map(x -> x >= 0, α_domain)))
  else
    domain_valid = (all(isfinite.(α_domain)))
  end

  if !domain_valid
    _min, _max = extrema(α_domain)
    error("Invalid $name in the domain extrema -> $((_min, _max))")
  end

  for axis in 1:N
    bc = haloedge_regions(full, axis, nhalo)
    lo_edge = bc.halo.lo
    hi_edge = bc.halo.hi

    α_lo = @view α[lo_edge]
    α_hi = @view α[hi_edge]

    if enforce_positivity
      α_lo_valid = (all(isfinite.(α_lo)) && all(map(x -> x >= 0, α_lo)))
      α_hi_valid = (all(isfinite.(α_hi)) && all(map(x -> x >= 0, α_hi)))
    else
      α_lo_valid = all(isfinite.(α_lo))
      α_hi_valid = all(isfinite.(α_hi))
    end

    if !α_lo_valid
      error("Invalid $name in the lo halo region for axis: $axis")
    end

    if !α_hi_valid
      error("Invalid $name in the hi halo region for axis: $axis")
    end
  end
end