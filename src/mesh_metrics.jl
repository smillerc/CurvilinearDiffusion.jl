"""Update the mesh metrics. Only do this whenever the mesh moves"""
function update_mesh_metrics!(solver, mesh::CurvilinearGrid2D)
  @unpack ilo, ihi, jlo, jhi = mesh.limits

  if solver.conservative
    @inline for idx in solver.domain_indices
      solver.J[idx] = jacobian(mesh, idx.I)
      solver.metrics[idx] = _conservative_metrics(mesh, idx.I)
    end
  else
    @inline for idx in solver.domain_indices
      solver.J[idx] = jacobian(mesh, idx.I)
      solver.metrics[idx] = _non_conservative_metrics(mesh, idx.I)
    end
  end

  return nothing
end

@inline function _non_conservative_metrics_2d(mesh, (i, j)::NTuple{2,Real})
  # note, metrics(mesh, (i,j)) uses node-based indexing, and here we're making
  # a tuple of metrics that uses cell-based indexing, thus the weird 1/2 offsets
  metricsᵢⱼ = cell_metrics(mesh, (i, j))
  metricsᵢ₊½ = cell_metrics(mesh, (i + 1 / 2, j))
  metricsⱼ₊½ = cell_metrics(mesh, (i, j + 1 / 2))
  metricsᵢ₋½ = cell_metrics(mesh, (i - 1 / 2, j))
  metricsⱼ₋½ = cell_metrics(mesh, (i, j - 1 / 2))
  # metricsᵢⱼ = metrics(mesh, (i + 1 / 2, j + 1 / 2))
  # metricsᵢ₋½ = metrics(mesh, (i, j + 1 / 2))
  # metricsⱼ₋½ = metrics(mesh, (i + 1 / 2, j))
  # metricsᵢ₊½ = metrics(mesh, (i + 1, j + 1 / 2))
  # metricsⱼ₊½ = metrics(mesh, (i + 1 / 2, j + 1))

  return (
    ξx=metricsᵢⱼ.ξx,
    ξy=metricsᵢⱼ.ξy,
    ηx=metricsᵢⱼ.ηx,
    ηy=metricsᵢⱼ.ηy,
    ξxᵢ₋½=metricsᵢ₋½.ξx,
    ξyᵢ₋½=metricsᵢ₋½.ξy,
    ηxᵢ₋½=metricsᵢ₋½.ηx,
    ηyᵢ₋½=metricsᵢ₋½.ηy,
    ξxⱼ₋½=metricsⱼ₋½.ξx,
    ξyⱼ₋½=metricsⱼ₋½.ξy,
    ηxⱼ₋½=metricsⱼ₋½.ηx,
    ηyⱼ₋½=metricsⱼ₋½.ηy,
    ξxᵢ₊½=metricsᵢ₊½.ξx,
    ξyᵢ₊½=metricsᵢ₊½.ξy,
    ηxᵢ₊½=metricsᵢ₊½.ηx,
    ηyᵢ₊½=metricsᵢ₊½.ηy,
    ξxⱼ₊½=metricsⱼ₊½.ξx,
    ξyⱼ₊½=metricsⱼ₊½.ξy,
    ηxⱼ₊½=metricsⱼ₊½.ηx,
    ηyⱼ₊½=metricsⱼ₊½.ηy,
  )
end

# construct the 1D conservative metrics
@inline function _conservative_metrics(mesh, (i,)::NTuple{1,Real})
  mᵢ₊½ = metrics(mesh, i + 1)
  mᵢⱼ₋½ = metrics(mesh, i)

  Jᵢ₊½ = mᵢ₊½.J
  Jⱼ₊½ = mⱼ₊½.J

  Jξx_ᵢ₊½ = mᵢ₊½.ξx * mᵢ₊½.J
  Jξx_ᵢ₋½ = mᵢⱼ₋½.ξx * mᵢⱼ₋½.J

  return (; Jᵢ₊½, Jⱼ₊½, Jξx_ᵢ₊½, Jξx_ᵢ₋½)
end

# construct the 2D conservative metrics
@inline function _conservative_metrics(mesh, (i, j)::NTuple{2,Real})
  mᵢ₊½ = metrics(mesh, (i + 1, j))
  mⱼ₊½ = metrics(mesh, (i, j + 1))
  mᵢⱼ₋½ = metrics(mesh, (i, j))

  Jᵢ₊½ = mᵢ₊½.J
  Jⱼ₊½ = mⱼ₊½.J
  Jᵢ₋½ = mᵢⱼ₋½.J
  Jⱼ₋½ = mᵢⱼ₋½.J

  # i + 1/2
  Jξx_ᵢ₊½ = mᵢ₊½.ξx * mᵢ₊½.J
  Jξy_ᵢ₊½ = mᵢ₊½.ξy * mᵢ₊½.J
  Jηx_ᵢ₊½ = mᵢ₊½.ηx * mᵢ₊½.J
  Jηy_ᵢ₊½ = mᵢ₊½.ηy * mᵢ₊½.J

  # i - 1/2, j - 1/2
  Jξx_ᵢ₋½ = Jξx_ⱼ₋½ = mᵢⱼ₋½.ξx * mᵢⱼ₋½.J
  Jξy_ᵢ₋½ = Jξy_ⱼ₋½ = mᵢⱼ₋½.ξy * mᵢⱼ₋½.J
  Jηx_ᵢ₋½ = Jηx_ⱼ₋½ = mᵢⱼ₋½.ηx * mᵢⱼ₋½.J
  Jηy_ᵢ₋½ = Jηy_ⱼ₋½ = mᵢⱼ₋½.ηy * mᵢⱼ₋½.J

  # j + 1/2
  Jξx_ⱼ₊½ = mⱼ₊½.ξx * mⱼ₊½.J
  Jξy_ⱼ₊½ = mⱼ₊½.ξy * mⱼ₊½.J
  Jηx_ⱼ₊½ = mⱼ₊½.ηx * mⱼ₊½.J
  Jηy_ⱼ₊½ = mⱼ₊½.ηy * mⱼ₊½.J

  return (;
    Jᵢ₊½,
    Jⱼ₊½,
    Jᵢ₋½,
    Jⱼ₋½,
    Jξx_ᵢ₊½,
    Jξy_ᵢ₊½,
    Jηx_ᵢ₊½,
    Jηy_ᵢ₊½,
    Jξx_ᵢ₋½,
    Jξy_ᵢ₋½,
    Jηx_ᵢ₋½,
    Jηy_ᵢ₋½,
    Jξx_ⱼ₊½,
    Jξy_ⱼ₊½,
    Jηx_ⱼ₊½,
    Jηy_ⱼ₊½,
    Jξx_ⱼ₋½,
    Jξy_ⱼ₋½,
    Jηx_ⱼ₋½,
    Jηy_ⱼ₋½,
  )
end

# construct the 3D conservative metrics
@inline function _conservative_metrics(mesh, (i, j, k)::NTuple{3,Real})
  mᵢ₊½ = metrics(mesh, (i + 1, j, k))
  mⱼ₊½ = metrics(mesh, (i, j + 1, k))
  mₖ₊½ = metrics(mesh, (i, j, k + 1))
  mᵢⱼₖ₋½ = metrics(mesh, (i, j, k))

  Jᵢ₊½ = mᵢ₊½.J
  Jⱼ₊½ = mⱼ₊½.J
  Jₖ₊½ = mⱼ₊½.J
  Jᵢ₋½ = mᵢⱼₖ₋½.J
  Jⱼ₋½ = mᵢⱼₖ₋½.J
  Jₖ₋½ = mᵢⱼₖ₋½.J

  # i + 1/2
  Jξx_ᵢ₊½ = mᵢ₊½.ξx * mᵢ₊½.J
  Jξy_ᵢ₊½ = mᵢ₊½.ξy * mᵢ₊½.J
  Jξz_ᵢ₊½ = mᵢ₊½.ξz * mᵢ₊½.J
  Jηx_ᵢ₊½ = mᵢ₊½.ηx * mᵢ₊½.J
  Jηy_ᵢ₊½ = mᵢ₊½.ηy * mᵢ₊½.J
  Jηz_ᵢ₊½ = mᵢ₊½.ηz * mᵢ₊½.J
  Jζx_ᵢ₊½ = mᵢ₊½.ζx * mᵢ₊½.J
  Jζy_ᵢ₊½ = mᵢ₊½.ζy * mᵢ₊½.J
  Jζz_ᵢ₊½ = mᵢ₊½.ζz * mᵢ₊½.J

  # j + 1/2
  Jξx_ⱼ₊½ = mⱼ₊½.ξx * mⱼ₊½.J
  Jξy_ⱼ₊½ = mⱼ₊½.ξy * mⱼ₊½.J
  Jξz_ⱼ₊½ = mⱼ₊½.ξz * mⱼ₊½.J
  Jηx_ⱼ₊½ = mⱼ₊½.ηx * mⱼ₊½.J
  Jηy_ⱼ₊½ = mⱼ₊½.ηy * mⱼ₊½.J
  Jηz_ⱼ₊½ = mⱼ₊½.ηz * mⱼ₊½.J
  Jζx_ⱼ₊½ = mⱼ₊½.ζx * mⱼ₊½.J
  Jζy_ⱼ₊½ = mⱼ₊½.ζy * mⱼ₊½.J
  Jζz_ⱼ₊½ = mⱼ₊½.ζz * mⱼ₊½.J

  # k + 1/2
  Jξx_ₖ₊½ = mₖ₊½.ξx * mₖ₊½.J
  Jξy_ₖ₊½ = mₖ₊½.ξy * mₖ₊½.J
  Jξz_ₖ₊½ = mₖ₊½.ξz * mₖ₊½.J
  Jηx_ₖ₊½ = mₖ₊½.ηx * mₖ₊½.J
  Jηy_ₖ₊½ = mₖ₊½.ηy * mₖ₊½.J
  Jηz_ₖ₊½ = mₖ₊½.ηz * mₖ₊½.J
  Jζx_ₖ₊½ = mₖ₊½.ζx * mₖ₊½.J
  Jζy_ₖ₊½ = mₖ₊½.ζy * mₖ₊½.J
  Jζz_ₖ₊½ = mₖ₊½.ζz * mₖ₊½.J

  # i - 1/2, j - 1/2, k - 1/2
  Jξx_ᵢ₋½ = Jξx_ⱼ₋½ = Jξx_ₖ₋½ = mᵢⱼₖ₋½.ξx * mᵢⱼₖ₋½.J
  Jξy_ᵢ₋½ = Jξy_ⱼ₋½ = Jξy_ₖ₋½ = mᵢⱼₖ₋½.ξy * mᵢⱼₖ₋½.J
  Jξz_ᵢ₋½ = Jξz_ⱼ₋½ = Jξz_ₖ₋½ = mᵢⱼₖ₋½.ξz * mᵢⱼₖ₋½.J
  Jηx_ᵢ₋½ = Jηx_ⱼ₋½ = Jηx_ₖ₋½ = mᵢⱼₖ₋½.ηx * mᵢⱼₖ₋½.J
  Jηy_ᵢ₋½ = Jηy_ⱼ₋½ = Jηy_ₖ₋½ = mᵢⱼₖ₋½.ηy * mᵢⱼₖ₋½.J
  Jηz_ᵢ₋½ = Jηz_ⱼ₋½ = Jηz_ₖ₋½ = mᵢⱼₖ₋½.ηz * mᵢⱼₖ₋½.J
  Jζx_ᵢ₋½ = Jζx_ⱼ₋½ = Jζx_ₖ₋½ = mᵢⱼₖ₋½.ζx * mᵢⱼₖ₋½.J
  Jζy_ᵢ₋½ = Jζy_ⱼ₋½ = Jζy_ₖ₋½ = mᵢⱼₖ₋½.ζy * mᵢⱼₖ₋½.J
  Jζz_ᵢ₋½ = Jζz_ⱼ₋½ = Jζz_ₖ₋½ = mᵢⱼₖ₋½.ζz * mᵢⱼₖ₋½.J

  return (;
    Jᵢ₊½,
    Jⱼ₊½,
    Jₖ₊½,
    Jᵢ₋½,
    Jⱼ₋½,
    Jₖ₋½,
    Jξx_ᵢ₊½,
    Jξy_ᵢ₊½,
    Jξz_ᵢ₊½,
    Jηx_ᵢ₊½,
    Jηy_ᵢ₊½,
    Jηz_ᵢ₊½,
    Jζx_ᵢ₊½,
    Jζy_ᵢ₊½,
    Jζz_ᵢ₊½,
    Jξx_ᵢ₋½,
    Jξy_ᵢ₋½,
    Jξz_ᵢ₋½,
    Jηx_ᵢ₋½,
    Jηy_ᵢ₋½,
    Jηz_ᵢ₋½,
    Jζx_ᵢ₋½,
    Jζy_ᵢ₋½,
    Jζz_ᵢ₋½,
    Jξx_ⱼ₊½,
    Jξy_ⱼ₊½,
    Jξz_ⱼ₊½,
    Jηx_ⱼ₊½,
    Jηy_ⱼ₊½,
    Jηz_ⱼ₊½,
    Jζx_ⱼ₊½,
    Jζy_ⱼ₊½,
    Jζz_ⱼ₊½,
    Jξx_ⱼ₋½,
    Jξy_ⱼ₋½,
    Jξz_ⱼ₋½,
    Jηx_ⱼ₋½,
    Jηy_ⱼ₋½,
    Jηz_ⱼ₋½,
    Jζx_ⱼ₋½,
    Jζy_ⱼ₋½,
    Jζz_ⱼ₋½,
    Jξx_ₖ₊½,
    Jξy_ₖ₊½,
    Jξz_ₖ₊½,
    Jηx_ₖ₊½,
    Jηy_ₖ₊½,
    Jηz_ₖ₊½,
    Jζx_ₖ₊½,
    Jζy_ₖ₊½,
    Jζz_ₖ₊½,
    Jξx_ₖ₋½,
    Jξy_ₖ₋½,
    Jξz_ₖ₋½,
    Jηx_ₖ₋½,
    Jηy_ₖ₋½,
    Jηz_ₖ₋½,
    Jζx_ₖ₋½,
    Jζy_ₖ₋½,
    Jζz_ₖ₋½,
  )
end
