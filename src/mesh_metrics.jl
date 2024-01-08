"""Update the mesh metrics. Only do this whenever the mesh moves"""
function update_mesh_metrics!(solver, mesh::CurvilinearGrid2D)
  @unpack ilo, ihi, jlo, jhi = mesh.limits

  if solver.conservative
    @inline for j in jlo:jhi
      for i in ilo:ihi
        solver.J[i, j] = jacobian(mesh, (i, j))
        solver.metrics[i, j] = _conservative_metrics_2d(mesh, i, j)
      end
    end
  else
    @inline for j in jlo:jhi
      for i in ilo:ihi
        solver.J[i, j] = jacobian(mesh, (i, j))
        solver.metrics[i, j] = _non_conservative_metrics_2d(mesh, i, j)
      end
    end
  end

  return nothing
end

@inline function _non_conservative_metrics_2d(mesh, i, j)
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

@inline function _conservative_metrics_2d(mesh, i, j)
  metrics_i_plus_half = metrics_with_jacobian(mesh, (i + 1, j))
  metrics_i_minus_half = metrics_with_jacobian(mesh, (i, j))
  metrics_j_plus_half = metrics_with_jacobian(mesh, (i, j + 1))
  metrics_j_minus_half = metrics_with_jacobian(mesh, (i, j))
  # metrics_i_plus_half = metrics_with_jacobian(mesh, (i + 1 / 2, j))
  # metrics_i_minus_half = metrics_with_jacobian(mesh, (i - 1 / 2, j))
  # metrics_j_plus_half = metrics_with_jacobian(mesh, (i, j + 1 / 2))
  # metrics_j_minus_half = metrics_with_jacobian(mesh, (i, j - 1 / 2))

  return (
    Jᵢ₊½=metrics_i_plus_half.J,
    Jξx_ᵢ₊½=metrics_i_plus_half.ξx * metrics_i_plus_half.J,
    Jξy_ᵢ₊½=metrics_i_plus_half.ξy * metrics_i_plus_half.J,
    Jηx_ᵢ₊½=metrics_i_plus_half.ηx * metrics_i_plus_half.J,
    Jηy_ᵢ₊½=metrics_i_plus_half.ηy * metrics_i_plus_half.J,
    Jᵢ₋½=metrics_i_minus_half.J,
    Jξx_ᵢ₋½=metrics_i_minus_half.ξx * metrics_i_minus_half.J,
    Jξy_ᵢ₋½=metrics_i_minus_half.ξy * metrics_i_minus_half.J,
    Jηx_ᵢ₋½=metrics_i_minus_half.ηx * metrics_i_minus_half.J,
    Jηy_ᵢ₋½=metrics_i_minus_half.ηy * metrics_i_minus_half.J,
    Jⱼ₊½=metrics_j_plus_half.J,
    Jξx_ⱼ₊½=metrics_j_plus_half.ξx * metrics_j_plus_half.J,
    Jξy_ⱼ₊½=metrics_j_plus_half.ξy * metrics_j_plus_half.J,
    Jηx_ⱼ₊½=metrics_j_plus_half.ηx * metrics_j_plus_half.J,
    Jηy_ⱼ₊½=metrics_j_plus_half.ηy * metrics_j_plus_half.J,
    Jⱼ₋½=metrics_j_minus_half.J,
    Jξx_ⱼ₋½=metrics_j_minus_half.ξx * metrics_j_minus_half.J,
    Jξy_ⱼ₋½=metrics_j_minus_half.ξy * metrics_j_minus_half.J,
    Jηx_ⱼ₋½=metrics_j_minus_half.ηx * metrics_j_minus_half.J,
    Jηy_ⱼ₋½=metrics_j_minus_half.ηy * metrics_j_minus_half.J,
  )
end

"""
conservative_edge_terms(edge_diffusivity::NTuple{4,T}, m) where {T}

Collect and find the edge terms used for the conservative form of the diffusion equation.
These are essentially the edge diffusivity + grid metric terms
"""
@inline function conservative_edge_terms(
  edge_diffusivity::NTuple{4,T}, m::NamedTuple
) where {T}

  # m is a NamedTuple that contains the conservative edge metris for 
  # a single cell. The names should be self-explainatory
  @unpack αᵢ₊½, αᵢ₋½, αⱼ₊½, αⱼ₋½ = edge_diffusivity

  fᵢ₊½ = αᵢ₊½ * (m.Jξx_ᵢ₊½^2 + m.Jξy_ᵢ₊½^2) / m.Jᵢ₊½
  fᵢ₋½ = αᵢ₋½ * (m.Jξx_ᵢ₋½^2 + m.Jξy_ᵢ₋½^2) / m.Jᵢ₋½
  fⱼ₊½ = αⱼ₊½ * (m.Jηx_ⱼ₊½^2 + m.Jηy_ⱼ₊½^2) / m.Jⱼ₊½
  fⱼ₋½ = αⱼ₋½ * (m.Jηx_ⱼ₋½^2 + m.Jηy_ⱼ₋½^2) / m.Jⱼ₋½
  gᵢ₊½ = αᵢ₊½ * (m.Jξx_ᵢ₊½ * m.Jηx_ᵢ₊½ + m.Jξy_ᵢ₊½ * m.Jηy_ᵢ₊½) / (4m.Jᵢ₊½)
  gᵢ₋½ = αᵢ₋½ * (m.Jξx_ᵢ₋½ * m.Jηx_ᵢ₋½ + m.Jξy_ᵢ₋½ * m.Jηy_ᵢ₋½) / (4m.Jᵢ₋½)
  gⱼ₊½ = αⱼ₊½ * (m.Jξx_ⱼ₊½ * m.Jηx_ⱼ₊½ + m.Jξy_ⱼ₊½ * m.Jηy_ⱼ₊½) / (4m.Jⱼ₊½)
  gⱼ₋½ = αⱼ₋½ * (m.Jξx_ⱼ₋½ * m.Jηx_ⱼ₋½ + m.Jξy_ⱼ₋½ * m.Jηy_ⱼ₋½) / (4m.Jⱼ₋½)

  return (; fᵢ₊½, fᵢ₋½, fⱼ₊½, fⱼ₋½, gᵢ₊½, gᵢ₋½, gⱼ₊½, gⱼ₋½)
end