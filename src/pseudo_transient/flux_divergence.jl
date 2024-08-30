"""
    flux_divergence(q, mesh, idx)

Compute the divergence of the flux, e.g. ∇⋅(α∇H), where the flux is `q = α∇H`
"""
function flux_divergence((qᵢ, qⱼ), (αᵢⱼ, βᵢⱼ), cell_center_metrics, idx::CartesianIndex{2})
  @inbounds begin
    i, j = idx.I

    ξx = cell_center_metrics.ξ.x₁[i, j]
    ξy = cell_center_metrics.ξ.x₂[i, j]
    ηx = cell_center_metrics.η.x₁[i, j]
    ηy = cell_center_metrics.η.x₂[i, j]

    _∂qᵢ∂ξ = qᵢ[i, j] - qᵢ[i - 1, j]
    _∂qⱼ∂η = qⱼ[i, j] - qⱼ[i, j - 1]
    _∂qᵢ∂ξ = _∂qᵢ∂ξ * !isapprox(qᵢ[i, j], qᵢ[i - 1, j])
    _∂qⱼ∂η = _∂qⱼ∂η * !isapprox(qⱼ[i, j], qⱼ[i, j - 1])

    ∂qᵢ∂ξ = (ξx^2 + ξy^2) * _∂qᵢ∂ξ
    ∂qⱼ∂η = (ηx^2 + ηy^2) * _∂qⱼ∂η

    ∂qᵢ∂η =
      0.25(ηx * ξx + ηy * ξy) * (
        (qᵢ[i, j + 1] + qᵢ[i - 1, j + 1]) - # take average on either side
        (qᵢ[i, j - 1] + qᵢ[i - 1, j - 1])   # and do diff in j
      )

    ∂qⱼ∂ξ =
      0.25(ηx * ξx + ηy * ξy) * (
        (qⱼ[i + 1, j] + qⱼ[i + 1, j - 1]) - # take average on either side
        (qⱼ[i - 1, j] + qⱼ[i - 1, j - 1])   # and do diff in i
      )

    ∂H∂ξ = aᵢⱼ[i, j] * 0.5(qᵢ[i, j] + qᵢ[i - 1, j]) # ∂u/∂ξ + non-orth terms
    ∂H∂η = bᵢⱼ[i, j] * 0.5(qⱼ[i, j] + qⱼ[i, j - 1]) # ∂u/∂η + non-orth terms
  end

  ∇q = ∂qᵢ∂ξ + ∂qⱼ∂η + ∂qᵢ∂η + ∂qⱼ∂ξ + ∂H∂ξ + ∂H∂η
  return ∇q
end

@inline function flux_divergence(
  (qᵢ, qⱼ, qₖ), (αᵢⱼₖ, βᵢⱼₖ, γᵢⱼₖ), cell_center_metrics, idx::CartesianIndex{3}
)
  @inbounds begin
    i, j, k = idx.I

    ξx = cell_center_metrics.ξ.x₁[idx]
    ξy = cell_center_metrics.ξ.x₂[idx]
    ξz = cell_center_metrics.ξ.x₃[idx]

    ηx = cell_center_metrics.η.x₁[idx]
    ηy = cell_center_metrics.η.x₂[idx]
    ηz = cell_center_metrics.η.x₃[idx]

    ζx = cell_center_metrics.ζ.x₁[idx]
    ζy = cell_center_metrics.ζ.x₂[idx]
    ζz = cell_center_metrics.ζ.x₃[idx]

    _∂qᵢ∂ξ = qᵢ[i, j, k] - qᵢ[i - 1, j, k]
    _∂qⱼ∂η = qⱼ[i, j, k] - qⱼ[i, j - 1, k]
    _∂qₖ∂ζ = qₖ[i, j, k] - qₖ[i, j, k - 1]

    _∂qᵢ∂ξ = _∂qᵢ∂ξ * !isapprox(qᵢ[i, j, k], qᵢ[i - 1, j, k])
    _∂qⱼ∂η = _∂qⱼ∂η * !isapprox(qⱼ[i, j, k], qⱼ[i, j - 1, k])
    _∂qₖ∂ζ = _∂qₖ∂ζ * !isapprox(qₖ[i, j, k], qₖ[i, j, k - 1])

    ∂qᵢ∂ξ = (ξx^2 + ξy^2 + ξz^2) * _∂qᵢ∂ξ
    ∂qⱼ∂η = (ηx^2 + ηy^2 + ηz^2) * _∂qⱼ∂η
    ∂qₖ∂ζ = (ζx^2 + ζy^2 + ζz^2) * _∂qₖ∂ζ

    # ---------------
    # ∂/∂η (α ∂u/∂ξ), aka ∂qᵢ/∂η
    # inner index is i  , i-1
    # outer index is j-1, j+1
    ∂qᵢ∂η =
      0.5(ηx * ξx + ηy * ξy + ηz * ξz) * (
        #  take average and do diff in j (for ∂/∂η)
        0.5(qᵢ[i, j + 1, k] + qᵢ[i - 1, j + 1, k]) - # j + 1
        0.5(qᵢ[i, j - 1, k] + qᵢ[i - 1, j - 1, k])   # j - 1
      )

    # ∂/∂ξ (α ∂u/∂η), aka ∂qⱼ/∂ξ
    # inner index is j  , j-1
    # outer index is i-1, i+1
    ∂qⱼ∂ξ =
      0.5(ηx * ξx + ηy * ξy + ηz * ξz) * (
        #  take average and do diff in i (for ∂/∂ξ)
        0.5(qⱼ[i + 1, j, k] + qⱼ[i + 1, j - 1, k]) - # i + 1
        0.5(qⱼ[i - 1, j, k] + qⱼ[i - 1, j - 1, k])   # i - 1
      )

    # # ---------------

    # ∂/∂ζ (α ∂u/∂η), aka ∂qⱼ/∂ζ
    # inner index is j  , j-1
    # outer index is k-1, k+1
    ∂qⱼ∂ζ =
      0.5(ζx * ηx + ζy * ηy + ζz * ηz) * (
        #  take average and do diff in k (for ∂/∂ζ)
        0.5(qⱼ[i, j, k + 1] + qⱼ[i, j - 1, k + 1]) - # k + 1
        0.5(qⱼ[i, j, k - 1] + qⱼ[i, j - 1, k - 1])   # k - 1
      )

    # ∂/∂η (α ∂u/∂ζ), aka ∂qₖ/∂η
    # inner index is k  , k-1
    # outer index is j-1, j+1
    ∂qₖ∂η =
      0.5(ζx * ηx + ζy * ηy + ζz * ηz) * (
        #  take average and do diff in j (for ∂/∂η)
        0.5(qₖ[i, j + 1, k] + qₖ[i, j + 1, k - 1]) - # j + 1
        0.5(qₖ[i, j - 1, k] + qₖ[i, j - 1, k - 1])   # j - 1
      )

    # # ---------------

    # ∂/∂ζ (α ∂u/∂ξ), aka ∂qᵢ/∂ζ
    # inner index is i  , i-1
    # outer index is k-1, k+1
    ∂qᵢ∂ζ =
      0.5(ζx * ξx + ζy * ξy + ζz * ξz) * (
        #  take average and do diff in k (for ∂/∂ζ)
        0.5(qᵢ[i, j, k + 1] + qᵢ[i - 1, j, k + 1]) - # k + 1
        0.5(qᵢ[i, j, k - 1] + qᵢ[i - 1, j, k - 1])   # k - 1
      )

    # ∂/∂ξ (α ∂u/∂ζ), aka ∂qₖ/∂ξ
    # inner index is k  , k-1
    # outer index is i-1, i+1
    ∂qₖ∂ξ =
      0.5(ζx * ξx + ζy * ξy + ζz * ξz) * (
        #  take average and do diff in i (for ∂/∂ξ)
        0.5(qₖ[i + 1, j, k] + qₖ[i + 1, j, k - 1]) - # i + 1
        0.5(qₖ[i - 1, j, k] + qₖ[i - 1, j, k - 1])   # i - 1
      )

    # ---------------

    # additional non-orthogonal terms
    ∂q∂ξ_α = αᵢⱼₖ[i, j, k] * 0.5(qᵢ[i, j, k] + qᵢ[i - 1, j, k])
    ∂q∂η_β = βᵢⱼₖ[i, j, k] * 0.5(qⱼ[i, j, k] + qⱼ[i, j - 1, k])
    ∂q∂ζ_γ = γᵢⱼₖ[i, j, k] * 0.5(qₖ[i, j, k] + qₖ[i, j, k - 1])
  end

  ∇q = (
    ∂qᵢ∂ξ +
    ∂qⱼ∂η +
    ∂qₖ∂ζ +
    #
    ∂qᵢ∂η +
    ∂qᵢ∂ζ +
    ∂qⱼ∂ξ +
    ∂qⱼ∂ζ +
    ∂qₖ∂η +
    ∂qₖ∂ξ +
    #
    ∂q∂ξ_α +
    ∂q∂η_β +
    ∂q∂ζ_γ
  )
  return ∇q
end

@inline function flux_divergence_orth(qξᵢ₊½, qξᵢ₋½, ξx)
  _dqξ = (qξᵢ₊½ - qξᵢ₋½)
  _dqξ = _dqξ * !isapprox(qξᵢ₊½, qξᵢ₋½)

  ∇q = (ξx^2) * _dqξ

  return ∇q
end

@inline function flux_divergence_orth(qξᵢ₊½, qξᵢ₋½, qηⱼ₊½, qηⱼ₋½, ξx, ξy, ηx, ηy)
  _dqξ = (qξᵢ₊½ - qξᵢ₋½)
  _dqη = (qηⱼ₊½ - qηⱼ₋½)
  _dqξ = _dqξ * !isapprox(qξᵢ₊½, qξᵢ₋½)
  _dqη = _dqη * !isapprox(qηⱼ₊½, qηⱼ₋½)

  ∂qξ∂ξ = (ξx^2 + ξy^2) * _dqξ
  ∂qη∂η = (ηx^2 + ηy^2) * _dqη

  ∇q = ∂qξ∂ξ + ∂qη∂η

  return ∇q
end

@inline function flux_divergence_orth(
  qξᵢ₊½, qξᵢ₋½, qηⱼ₊½, qηⱼ₋½, qζₖ₊½, qζₖ₋½, ξx, ξy, ξz, ηx, ηy, ηz, ζx, ζy, ζz
) where {T}
  _dqξ = qξᵢ₊½ - qξᵢ₋½
  _dqη = qηⱼ₊½ - qηⱼ₋½
  _dqζ = qζₖ₊½ - qζₖ₋½

  _dqξ = _dqξ * !isapprox(qξᵢ₊½, qξᵢ₋½)
  _dqη = _dqη * !isapprox(qηⱼ₊½, qηⱼ₋½)
  _dqζ = _dqζ * !isapprox(qζₖ₊½, qζₖ₋½)

  ∂qξ∂ξ = (ξx^2 + ξy^2 + ξz^2) * _dqξ
  ∂qη∂η = (ηx^2 + ηy^2 + ηz^2) * _dqη
  ∂qζ∂ζ = (ζx^2 + ζy^2 + ζz^2) * _dqζ

  ∇q = ∂qξ∂ξ + ∂qη∂η + ∂qζ∂ζ

  return ∇q
end